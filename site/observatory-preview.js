/* Explicit local fixtures. This file never starts a browser, calls an AI, or trains a model. */
(function () {
  "use strict";
  const stamp = () => new Date().toISOString();
  const clone = value => JSON.parse(JSON.stringify(value));
  const defaultObjective = "Research consciousness and subjective experience across neuroscience, psychology, philosophy of mind, reality and metaphysics, and religious and contemplative traditions, alongside AI consciousness, sentience, pain and moral patienthood. Compare original sources and competing interpretations, distinguishing empirical findings, philosophical arguments and religious interpretations.";
  const sources = [
    {id:"preview-butlin", title:"Consciousness in Artificial Intelligence: Insights from the Science of Consciousness", authors:"Patrick Butlin et al.", canonical_url:"https://arxiv.org/abs/2308.08708v3", version:"arXiv:2308.08708v3", agent_id:"scholar", source_type:"paper", license:"CC-BY-NC-SA-4.0", license_verified:false, rights_status:"reference_only", review_status:"reference_only", curation:{eligible:false,status:"reference_only",reasons:["license_not_in_public_corpus_policy"]}, summary:"A theory-derived set of computational indicators. The authors distinguish proposed indicators from a settled diagnostic test and assess systems available in 2023.", limitation:"The indicators depend on contested theories and computational assumptions. A noncommercial, share-alike license is outside the default permissive corpus policy.", rights_evidence:"The license link for the linked arXiv version displays CC BY-NC-SA 4.0. Separate permission or a compatible licensed copy is required.", topics:["indicators","global workspace","higher-order processing"], bookmarked:false},
    {id:"preview-chalmers", title:"Could a Large Language Model be Conscious?", authors:"David J. Chalmers", canonical_url:"https://arxiv.org/abs/2303.07103v3", version:"arXiv:2303.07103v3", agent_id:"cartographer", source_type:"paper", license:"arXiv distribution license", license_verified:false, rights_status:"reference_only", review_status:"reference_only", curation:{eligible:false,status:"reference_only",reasons:["license_not_in_public_corpus_policy"]}, summary:"A philosophical examination of obstacles and possible future candidates. The distinctions help organize reasons for and against consciousness without establishing a test.", limitation:"Philosophical argument is separate from measured evidence. The arXiv distribution license does not establish this project's full-text reuse permission.", rights_evidence:"The linked copy uses arXiv's nonexclusive distribution license. Keep a metadata reference unless separate permission is recorded.", topics:["philosophy","language models","uncertainty"], bookmarked:false},
    {id:"preview-keeling", title:"Can LLMs make trade-offs involving stipulated pain and pleasure states?", authors:"Geoffrey Keeling et al.", canonical_url:"https://arxiv.org/abs/2411.02432v1", version:"arXiv:2411.02432v1", agent_id:"sentinel", source_type:"paper", license:"CC-BY-4.0", license_verified:true, rights_status:"license_verified", review_status:"approved", curation:{eligible:true,status:"eligible",reasons:[]}, summary:"Textually stipulated penalties and rewards can change model choices. Instruction following, learned associations and task framing remain alternative explanations.", limitation:"Stipulated pain is a prompt condition. A change in choice does not demonstrate felt pain.", rights_evidence:"Example review: the linked arXiv copy displays CC BY 4.0; attribution and any separately credited material still need review before export.", topics:["behavior","valence","task framing"], bookmarked:false},
    {id:"preview-shardlow", title:"Deanthropomorphising NLP: Can a language model be conscious?", authors:"Matthew Shardlow and Piotr Przybyła", canonical_url:"https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307521", version:"Publisher version of record", agent_id:"skeptic", source_type:"paper", license:"CC-BY-4.0", license_verified:true, rights_status:"license_verified", review_status:"approved", curation:{eligible:true,status:"eligible",reasons:[]}, summary:"An architecture-based skeptical argument using Integrated Information Theory, with a critique of anthropomorphic descriptions of language model behavior.", limitation:"The conclusion depends on a contested consciousness theory and an architectural interpretation.", rights_evidence:"Example review: the PLOS publisher copy displays CC BY 4.0. Its publisher provenance is distinct from an arXiv copy with different terms.", topics:["skepticism","architecture","integrated information"], bookmarked:false},
    {id:"preview-seth", title:"Conscious artificial intelligence and biological naturalism", authors:"Anil K. Seth", canonical_url:"https://sussex.figshare.com/articles/journal_contribution/Conscious_artificial_intelligence_and_biological_naturalism/28649519", version:"Institutional repository copy", agent_id:"archivist", source_type:"paper", license:"CC-BY-4.0", license_verified:false, rights_status:"needs_review", review_status:"pending", curation:{eligible:false,status:"quarantined",reasons:["rights_not_verified","missing_rights_evidence"]}, summary:"A competing substrate-based perspective: properties of living organisms may matter beyond computation alone.", limitation:"Review the exact downloaded version and any separately credited material. Biological naturalism is a theoretical position.", rights_evidence:"", topics:["biology","substrate","computational sufficiency"], bookmarked:false}
  ];
  const roles = [
    ["scholar","The Scholar","Science of experience","Primary research on human, animal and artificial consciousness, neuroscience, psychology and computational theories; seek original studies and distinguish measurements from interpretations."],
    ["skeptic","The Skeptic","Contrary evidence","Alternative explanations, failed replications and objections to scientific, philosophical and religious claims about consciousness, including machine sentience; challenge attractive claims respectfully."],
    ["sentinel","The Sentinel","Welfare & measurement","Pain, suffering, welfare, moral patienthood and experimental measurement in humans, animals and AI; compare ethical arguments while distinguishing behavior, testimony and subjective experience."],
    ["cartographer","The Cartographer","Mind, reality & traditions","Map theories of mind, reality and metaphysics; compare phenomenology, religious and contemplative accounts across traditions and identify disagreements, terminology, gaps and explicit connections to consciousness and AI."],
    ["archivist","The Archivist","Rights & provenance","Find original source versions, article licensing statements and trustworthy provenance; unknown rights stay unverified."],
    ["curator","The Curator","Corpus quality","Investigate coverage gaps and source quality across the mission; separate empirical findings, philosophical arguments and religious interpretations, and draft useful instruction examples; drafts need independent owner review."]
  ];
  function create() {
    const now = stamp();
    const state = {
      demo:true, mission:{id:"preview-mission",status:"ready",objective:defaultObjective,until_stopped:true,open_questions:4},
      agents:roles.map((role,i)=>({id:role[0],name:role[1],role:role[2],specialty:role[3],status:"preview_ready",current_url:sources[i%sources.length].canonical_url,source_id:sources[i%sources.length].id,goal:i===0?"Compare indicator frameworks with the assumptions that support them.":role[3],last_action:"Example page inspection",last_observation:sources[i%sources.length].limitation,step:0})),
      sources:clone(sources).map((source,i)=>({...source,created_at:now,provenance:{method:"preview_fixture",collected_at:now,collector_version:"concept-preview"},content_hash:"preview-source-"+i,word_count:null})),
      notes:[
        {id:"preview-note-1",agent_id:"scholar",source_id:"preview-butlin",type:"observation",text:"Compare each proposed indicator with the assumptions of its source theory. Avoid treating the checklist as an accepted diagnostic.",support_verified:false,generated_by:"preview_fixture",review_status:"pending",bookmarked:false,created_at:now},
        {id:"preview-note-2",agent_id:"skeptic",source_id:"preview-shardlow",type:"question",text:"How does the architectural argument change if the consciousness theory changes?",support_verified:false,generated_by:"preview_fixture",review_status:"pending",bookmarked:false,created_at:now},
        {id:"preview-note-3",agent_id:"sentinel",source_id:"preview-keeling",type:"observation",text:"The experiment stipulates pain in text. Record instruction following as a competing explanation of the observed choice.",support_verified:false,generated_by:"preview_fixture",review_status:"pending",bookmarked:true,created_at:now}
      ],
      datasets:[],jobs:[],checkpoints:[{id:"preview-baseline",name:"Original chamber baseline",base_model:"Existing chamber model",status:"baseline",baseline:true,revision:"Preserved by the chamber",calibration:{status:"existing_baseline",passed:false},created_at:now}],
      events:[
        {id:1,seq:1,type:"preview.loaded",message:"Loaded illustrative source records. No browsing or training is running.",created_at:now},
        {id:2,seq:2,type:"agent.decision",agent_id:"scholar",message:"Compare indicator frameworks with their underlying assumptions.",data:{source_id:"preview-butlin"},created_at:now},
        {id:3,seq:3,type:"source.reviewed",agent_id:"archivist",message:"Example policy excludes the noncommercial and distribution-only copies from the permissive corpus.",created_at:now}
      ],
      connections:[{id:"research",name:"Research brain",status:"preview",message:"Simulated; no API calls"},{id:"browser",name:"Browser infrastructure",status:"preview",message:"Local illustrative article"},{id:"huggingface",name:"Hugging Face",status:"disconnected",message:"No uploads in preview"},{id:"training",name:"GPU worker",status:"disconnected",message:"No job allocated"}],
      settings:{research_provider:"x402",research_model:"simulated/openai-research",research_protocol:"responses",browser_provider:"local",hf_namespace:"",agent_count:6,hf_base_model:"meta-llama/Llama-3.1-70B",hf_base_revision:"349b2ddb53ce8f2849a6c168a81980ab25258dac",training_mode:"qlora",training_enabled:false,training_continue_from_previous:true,publish_policy:"private",synthetic_training_approved:false,provider_policy_reference:""},
      research_catalog:{status:"simulated",models:[{id:"simulated/openai-research",capability_source:"owner_declared",eligible:true,name:"Simulated OpenAI research model",protocols:["responses"],categories:["research"],capabilities:{vision:true,structured_actions:true,structured_outputs:true}},{id:"simulated/anthropic-research",capability_source:"owner_declared",eligible:true,name:"Simulated Anthropic research model",protocols:["messages"],categories:["research"],capabilities:{vision:true,structured_actions:true,tool_calling:true}}]},
      funding:{demo:true,status:"simulated",configured:false,enabled:false,wallet_address:null,network:null,asset:null,balance_atomic:"250000000",reserved_atomic:"3000000",settled_atomic:"12000000",daily_spent_and_reserved_atomic:"15000000",daily_remaining_atomic:"85000000",required_request_reserve_atomic:"5000000",receipts:[{request_id:"preview-payment-1",status:"simulated",amount_atomic:"12000000",transaction:null,network:null,asset:null}]},
      cursor:3,preview_step:0
    };
    return state;
  }
  function event(state,type,message,agent_id,data) {
    const item={id:++state.cursor,seq:state.cursor,type,message,agent_id,data:data||{},created_at:stamp()};
    state.events.push(item);
    if(state.events.length>100) state.events.shift();
    return item;
  }
  function step(state,selectedAgent) {
    const n=state.preview_step++, agent=state.agents.find(item=>item.id===selectedAgent) || state.agents[n%state.agents.length];
    const phase=agent.step%4, index=state.agents.indexOf(agent);
    const source=state.sources[(index+Math.floor(agent.step/4))%state.sources.length];
    const goals=[
      "Follow a citation to distinguish an empirical finding from its theoretical interpretation.",
      "Seek a counterargument or alternative explanation of this consciousness claim.",
      "Separate behavioral reports and testimony from evidence about subjective experience.",
      "Map how this account of mind or reality connects to consciousness and competing traditions.",
      "Check the license of this exact copy before proposing corpus inclusion.",
      "Compare this source with related versions and keep a source-linked uncertainty."
    ];
    agent.status=state.mission.status==="running"?"preview_browsing":"preview_ready";
    agent.current_url=source.canonical_url;agent.source_id=source.id;agent.goal=goals[n%goals.length];
    agent.preview_scroll_phase=phase;agent.preview_focus_note_id=null;
    const section=["summary","caveat","caveat","provenance"][phase];
    agent.preview_inspection_id="preview-inspection-"+agent.id+"-"+Math.floor(agent.step/4)+"-"+section;
    agent.last_action=["Example inspection of summary","Example scroll and inspection of caveat","Example note from selected passage","Example scroll and inspection of provenance"][phase];agent.step++;
    agent.last_observation=source.limitation;
    event(state,"agent.decision",agent.goal,agent.id,{source_id:source.id,step:agent.step});
    if(phase!==2)event(state,"agent.inspection","Example passage selected for inspection.",agent.id,{source_id:source.id,inspection_id:agent.preview_inspection_id});
    if(phase===2) {
      const note={id:"preview-note-"+state.cursor,agent_id:agent.id,source_id:source.id,type:"lead",text:source.limitation+" Follow-up: compare at least one competing interpretation.",passage:source.limitation,inspection_id:agent.preview_inspection_id,support_verified:false,generated_by:"preview_fixture",review_status:"pending",bookmarked:false,created_at:stamp()};
      state.notes.push(note);event(state,"note.saved","Example source-linked uncertainty saved.",agent.id,{source_id:source.id,note_id:note.id,inspection_id:agent.preview_inspection_id});
      agent.preview_focus_note_id=note.id;
    }
    state.mission.open_questions=4+Math.floor(n/6);
    return state;
  }
  function mission(state,action) {
    state.mission.status={start:"running",pause:"paused",resume:"running",stop:"stopped"}[action];
    state.agents.forEach(agent=>{agent.status=action==="stop"?"preview_stopped":action==="pause"?"preview_paused": "preview_browsing";});
    event(state,"mission."+action,"Simulated mission "+action+". No real worker was started.");
    return state;
  }
  function snapshot(state) {
    const eligible=state.sources.filter(source=>source.curation&&source.curation.eligible);
    const number=state.datasets.length+1, id="preview-snapshot-"+number;
    const manifest={demo:true,notice:"Illustrative metadata manifest. Contains no original full text or training records.",policy_version:"preview-permissive-corpus-v1",source_ids:eligible.map(source=>source.id),source_records:eligible.map(source=>({id:source.id,title:source.title,canonical_url:source.canonical_url,license:source.license,rights_evidence:source.rights_evidence,provenance:source.provenance,version:source.version})),excluded_sources:state.sources.filter(source=>!source.curation.eligible).map(source=>({source_id:source.id,reasons:source.curation.reasons})),synthetic_policy:{owner_approved:!!state.settings.synthetic_training_approved,provider_policy_reference:state.settings.provider_policy_reference||null}};
    const record={id,name:"Consciousness corpus · preview "+number,status:"preview_candidate",demo:true,immutable:true,source_ids:manifest.source_ids,manifest_hash:"example-fingerprint-"+number,manifest,created_at:stamp(),counts:{original_documents:eligible.length,train_documents:eligible.length,validation_documents:0,synthetic_examples:0,excluded_sources:manifest.excluded_sources.length}};
    state.datasets.push(record);event(state,"dataset.snapshot","Created a simulated metadata snapshot. No corpus was exported.",null,{snapshot_id:id});
    return record;
  }
  function train(state,snapshot_id,stage="cpt",parent_run_id=null) {
    const id="preview-job-"+(state.jobs.length+1), dataset=state.datasets.find(item=>item.id===snapshot_id);
    if(!dataset) throw new Error("Create a curated preview snapshot first.");
    const adaptation=({qlora:"QLoRA",lora:"LoRA"})[String(state.settings.training_mode || "qlora").toLowerCase()] || state.settings.training_mode;
    const record={id,name:"Preview job "+(state.jobs.length+1),snapshot_id,status:"preview_validated",stage,demo:true,created_at:stamp(),manifest:{base_model:state.settings.hf_base_model,base_revision:state.settings.hf_base_revision ?? null,training_mode:state.settings.training_mode,stage,parent_run_id,snapshot_id,snapshot_hash:dataset.manifest_hash},logs:["[preview] Simulate validation of the selected metadata snapshot.","[preview] Preserve the original baseline and source family splits.","[preview] Check rights decisions and exclude unapproved research commentary.","[preview] Selected recipe: "+state.settings.hf_base_model+" / "+adaptation+" / "+String(stage).toUpperCase()+".","[preview] Selected base revision: "+(state.settings.hf_base_revision || "resolve at preparation")+".","[preview] Validation flow complete. No GPU was allocated or model downloaded.","[preview] No weights, measured loss, or calibration result exists."],metrics:null};
    state.jobs.push(record);
    state.checkpoints.push({id:"preview-candidate-"+state.jobs.length,name:"Example candidate · awaiting weights",status:"preview_placeholder",demo:true,stage,base_model:state.settings.hf_base_model,base_revision:state.settings.hf_base_revision ?? null,run_id:id,snapshot_id,revision:"No weights created",checks:{passed:false},calibration:{passed:false,status:"required",checks:[{label:"Tokenizer and architecture compatibility",status:"not_measured"},{label:"New intervention vector",status:"not_measured"},{label:"Control comparisons",status:"not_measured"},{label:"Held-out evaluation",status:"not_measured"}]},created_at:stamp()});
    event(state,"training.preview","Validated a simulated job. Training did not run.",null,{run_id:id});
    return record;
  }
  window.ObservatoryPreview={create,event,step,mission,snapshot,train};
})();
