/* Actual UI helper checks against authored DOM/state fixtures. No browser/network. */
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),assert=require('node:assert/strict');
const main=fs.readFileSync(path.resolve(__dirname,'../../site/observatory.js'),'utf8');
let assertions=0;
function check(value,message){assert.ok(value,message);assertions++;}
function extract(name){
  const start=main.indexOf('  function '+name+'(');
  assert.ok(start>=0,'Actual UI helper exists: '+name);
  const next=main.slice(start+1).search(/\n  (?:async )?function \w+\(/);
  assert.ok(next>=0);return main.slice(start,start+1+next);
}
const nodes=new Map();
function $(id){if(!nodes.has(id))nodes.set(id,{value:'',checked:false,innerHTML:'',insertAdjacentHTML(position,html){this.innerHTML+=html;}});return nodes.get(id);}
const context={$,nodes,state:{mission:{status:'running'},agents:[],sources:[],settings:{}},mode:'connected',connected:true,
  busy:false,ownerToken:'fixture-owner',canMutate(){return context.connected && !!context.ownerToken;},
  toast(){},confirmAction(title,message,label,callback){context.confirmation={title,message,label,callback};},
  async mutate(route,payload){context.request={route,payload};return {acknowledged:true};},
  clearInheritedBaseRevision(){},compatibleResearchModels(){return [{id:'fixture-model'}];},
  sourceById(id){return context.state.sources.find(source=>source.id===id);}};
vm.createContext(context);
const helpers=['emptyState','normalizeState','titleCase','clock','curationReviews','curationDecision','sourceCurationReview',
  'automatedReviewBadge','curationRecoveryAvailable','curationRecoveryMarkup','reviewCurationRecovery','renderAutomatedCuration','fillAutoCurationSettings','collectSettings',
  'collectSourceReview','qualityReviewMarkup','qualityClassificationMarkup','syncSourceReviewFields','coverageBucketsMarkup','corpusAuditMarkup'];
const constants=main.split('\n').filter(line=>/^  const (esc|list|topicDomains|evidenceKinds|autoCurationPolicyAck) =/.test(line)).join('\n');
vm.runInContext(constants+'\n'+helpers.map(extract).join('\n'),context);
check(context.normalizeState({mission:{status:'stopped'},agents:[]}).curation_reviews.length===0,'Older backends have a safe empty review list');
check(context.normalizeState({mission:{status:'stopped'},agents:[],curation_reviews:{}}).curation_reviews.length===0,'Malformed optional receipts do not become UI records');
$('setting-provider').value='openai';$('setting-model').value='fixture-model';$('setting-base-model').value='fixture-base';
$('setting-browser').value='local';$('setting-protocol').value='responses';
let settings=context.collectSettings();
check(settings.auto_curation_enabled===false && settings.auto_curation_policy_ack==='','Review policy is not enabled by default');
$('setting-auto-curation').checked=true;settings=context.collectSettings();
check(settings.auto_curation_enabled===true && settings.auto_curation_policy_ack==='originals-v2','Opt-in sends the updated source-only classification policy');
check(settings.synthetic_training_approved===false,'Source review does not grant synthetic-output permission');
check(settings.training_enabled===false,'Source review does not enable paid GPU training');
$('setting-auto-curation').checked=false;settings=context.collectSettings();
check(settings.auto_curation_policy_ack==='','Disabling clears the policy acknowledgement in the request');
context.renderAutomatedCuration();
check($('automated-curation-summary').innerHTML.includes('Disabled'),'No receipts are presented as an active review');
context.state.settings.auto_curation_enabled=true;context.mode='preview';context.renderAutomatedCuration();
check($('automated-curation-summary').innerHTML.includes('Simulated review'),'Preview policy always remains simulated');
check($('automated-curation-summary').innerHTML.includes('No model call was made'),'Preview makes no real acceptance claim');
context.mode='connected';context.connected=false;context.renderAutomatedCuration();
check($('automated-curation-summary').innerHTML.includes('Backend unavailable'),'Enabled preference does not conceal disconnection');
context.connected=true;
context.state.settings={auto_curation_enabled:true,auto_curation_policy_ack:'originals-v1'};
context.renderAutomatedCuration();context.fillAutoCurationSettings(context.state.settings);
check($('automated-curation-summary').innerHTML.includes('Policy update required') && !$('automated-curation-summary').innerHTML.includes('Enabled during research'),'An obsolete acknowledgement cannot advertise an enabled worker');
check($('setting-auto-curation').checked===false && $('auto-curation-policy-notice').hidden===false,'Legacy opt-in needs an explicit updated choice');
context.state.settings.auto_curation_policy_ack='originals-v2';context.renderAutomatedCuration();context.fillAutoCurationSettings(context.state.settings);
check($('automated-curation-summary').innerHTML.includes('Enabled during research') && $('setting-auto-curation').checked===true && $('auto-curation-policy-notice').hidden===true,'The current acknowledgement enables classification during a running mission');
function reviewData(overrides={}) {
  const fields={review_status:'approved',license:'CC-BY-4.0',license_verified:'on',rights_evidence:'Exact copy license reviewed',quality_rationale:'Original argument is relevant; limits recorded',quality_reviewed_by:'Owner',topic_relevance:'relevant',topic_philosophy_of_mind:'on',evidence_kind:'philosophical_argument',evidence_stance:'not_applicable',quality_source_type:'theoretical_paper',...overrides};
  for(const [key,value] of Object.entries(fields))if(value===null)delete fields[key];
  return {get:key=>fields[key] ?? null,has:key=>Object.prototype.hasOwnProperty.call(fields,key)};
}
let reviewed=context.collectSourceReview(reviewData({covered_supportive:'on',covered_skeptical:'on'}));
check(reviewed.quality_review.topic_domains.join(',')==='philosophy_of_mind' && reviewed.quality_review.evidence_kind==='philosophical_argument','Actual review payload pairs research area and basis of claims');
check(reviewed.quality_review.evidence_stance==='not_applicable' && reviewed.quality_review.covered_stances.length===0,'General philosophy cannot acquire machine coverage through stale checked fields');
check(!Object.prototype.hasOwnProperty.call(reviewed,'source_type') && reviewed.quality_review.source_type==='theoretical_paper','Reviewed type does not overwrite collection provenance');
reviewed=context.collectSourceReview(reviewData({topic_religion_contemplation:'on',topic_philosophy_of_mind:null,evidence_kind:'religious_contemplative'}));
check(reviewed.quality_review.topic_domains[0]==='religion_contemplation' && reviewed.quality_review.covered_stances.length===0,'Contemplative claims are classified separately from scientific and machine evidence');
reviewed=context.collectSourceReview(reviewData({topic_machine_consciousness:'on',topic_consciousness_science:'on',evidence_kind:'empirical',evidence_stance:'mixed',covered_supportive:'on',covered_uncertain:'on'}));
check(reviewed.quality_review.topic_domains.includes('machine_consciousness') && reviewed.quality_review.covered_stances.join(',')==='supportive,uncertain','Mixed machine documents retain only reviewed machine perspectives');
for(const [overrides,message] of [[{topic_philosophy_of_mind:null},'research area'],[{evidence_kind:''},'basis'],[{evidence_stance:'skeptical'},'Not applicable'],[{rights_evidence:''},'Rights verification']]) {
  let blocked=false;try{context.collectSourceReview(reviewData(overrides));}catch(error){blocked=error.message.includes(message);}check(blocked,'Review payload fails visibly for missing or inconsistent '+message);
}
const fields=[{checked:true,disabled:false},{checked:true,disabled:false}], stance={value:'not_applicable',disabled:false},machine={checked:false};
const form={querySelector:selector=>selector.includes('evidence_stance')?stance:machine,querySelectorAll:()=>fields};
context.syncSourceReviewFields(form);
check(fields.every(field=>!field.checked && field.disabled),'Not-applicable stance clears and disables machine coverage controls');
stance.value='mixed';machine.checked=true;context.syncSourceReviewFields(form);check(fields.every(field=>!field.disabled),'Mixed machine review exposes covered perspectives');
fields[0].checked=true;machine.checked=false;context.syncSourceReviewFields(form,true);
check(stance.value==='not_applicable' && fields.every(field=>!field.checked && field.disabled),'Removing the machine area cannot leave machine perspectives selected');
const classification=context.qualityClassificationMarkup({quality_review:{status:'pending',topic_domains:['religion_contemplation'],evidence_kind:'religious_contemplative',evidence_stance:'not_applicable'}});
check(classification.includes('Religion &amp; contemplation') && classification.includes('Religious or contemplative interpretation') && classification.includes('Not applicable'),'Inspector shows the recorded research area, basis and machine applicability');
const reviewMarkup=context.qualityReviewMarkup({quality_review:{}},false);
check(reviewMarkup.includes('Machine-consciousness perspective') && reviewMarkup.includes('name="evidence_kind" disabled') && reviewMarkup.includes('name="topic_metaphysics_reality" disabled'),'Public inspector displays read-only paired classification fields');
check(context.qualityClassificationMarkup({}).includes('Not classified'),'Legacy reviews are not silently assigned new classification');
check(context.coverageBucketsMarkup(undefined,'Research areas',[])==='' && context.coverageBucketsMarkup({unknown:'4'},'Research areas',[])==='','Absent or invalid audit buckets never fabricate document counts');
check(context.coverageBucketsMarkup({religion_contemplation:3},'Research areas',[['religion_contemplation','Religion & contemplation']]).includes('3 documents'),'Coverage displays actual reported bucket counts');
check(!context.corpusAuditMarkup({coverage_audit:{splits:{train:{stances:{uncertain:2}}}},quality_gate:{ready:false}}).includes('Reviewed research areas'),'Legacy audits omit unmeasured scope buckets');
const source={id:'source-fixture',title:'<img src=x onerror=alert(1)>',review_status:'approved',
  curation:{eligible:true},quality_review:{reviewer_kind:'automated',status:'approved',model:'fixture-model'}};
context.state.sources=[source];context.state.curation_reviews=[{id:'receipt-fixture',source_id:source.id,
  decision:'accepted',model:'<script>unsafe</script>',rationale:'<img src=x onerror=alert(1)>',created_at:'2026-10-05T00:00:00Z'}];
context.renderAutomatedCuration();const rendered=$('automated-curation-summary').innerHTML;
check(rendered.includes('1</strong> accepted receipts'),'Counts identify historical receipts');
check(!rendered.includes('<script>') && !rendered.includes('<img src=x'),'Source and model content is escaped');
context.state.curation_reviews=[];
check(context.sourceCurationReview(source).decision==='accepted','A source with a bound approval can render without the optional receipt list');
context.state.curation_reviews=[{id:'receipt-fixture',source_id:source.id,decision:'accepted'}];
source.review_status='rejected';source.quality_review={status:'rejected',reviewed_by:'operator'};
check(context.automatedReviewBadge(source).includes('Prior automated acceptance'),'Operator rejection is not presented as current automated eligibility');
async function recoveryChecks(){
  context.state.curation_runtime={status:'reviewing',call_state:'awaiting_operator',active_stage:'critic',review_id:'review-recovery',source_id:source.id};
  check(!context.curationRecoveryAvailable('review-recovery'),'Running mission cannot authorize an interrupted retry');
  context.state.mission.status='faulted';context.renderAutomatedCuration();
  check($('automated-curation-summary').innerHTML.includes('Operator attention required'),'Interrupted paid review is visible');
  check(context.curationRecoveryAvailable('review-recovery'),'Idle owner can review a matching interrupted call');
  check(!context.curationRecoveryAvailable('changed-review'),'Stale review ID cannot authorize recovery');
  context.ownerToken='';check(!context.curationRecoveryAvailable('review-recovery'),'Public visitor cannot acknowledge recovery');context.ownerToken='fixture-owner';
  context.mode='preview';check(!context.curationRecoveryAvailable('review-recovery') && context.curationRecoveryMarkup(context.state.curation_runtime)==='','Preview never simulates recovery');context.mode='connected';
  context.reviewCurationRecovery('review-recovery');
  check(context.confirmation.message.includes('may already have been billed') && !context.request,'Another attempt requires explicit acknowledgement of billing uncertainty');
  context.state.curation_runtime.review_id='different';await context.confirmation.callback();
  check(!context.request,'Changed server state invalidates a confirmation');
  context.state.curation_runtime.review_id='review-recovery';context.reviewCurationRecovery('review-recovery');await context.confirmation.callback();
  check(context.request.route==='admin/curation/recovery' && context.request.payload.reviewed===true && context.request.payload.review_id==='review-recovery','Acknowledgement uses the exact owner-only recovery route');
  check(context.state.mission.status==='faulted','Acknowledgement does not resume research');
  console.log(JSON.stringify({automaticCurationUi:'passed',assertions,actualHelpers:true,sourceOnlyPolicy:true,operatorOverride:true,explicitRecovery:true,noNetwork:true}));
}
recoveryChecks().catch(error=>{console.error(error);process.exitCode=1;});
