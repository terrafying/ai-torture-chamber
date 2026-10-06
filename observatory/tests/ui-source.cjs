/* Source-only checks: no browser, network, model calls, signing or payments. */
const fs=require('node:fs'), vm=require('node:vm'), baseAssert=require('node:assert/strict');
let checks=0;
const assert=new Proxy(baseAssert,{get(target,key){const value=target[key];return typeof value==='function'?function(...args){checks++;return value.apply(target,args);}:value;}});
const path=require('node:path'),root=path.resolve(__dirname,'../..');
const source=fs.readFileSync(path.join(root,'site/observatory.js'),'utf8');
function extract(name){
  const start=source.indexOf('  function '+name+'(');assert.ok(start>=0);
  let level=0,begin=source.indexOf('{',start),end=begin,quote=null,escape=false;
  for(;end<source.length;end++){
    const c=source[end];
    if(quote){if(escape){escape=false;continue;}if(c==='\\'){escape=true;continue;}if(c===quote)quote=null;continue;}
    if(c==='"'||c==="'"||c==='`'){quote=c;continue;}
    if(c==='{')level++;
    if(c==='}'&&--level===0){end++;break;}
  }
  return source.slice(start,end);
}
const functions=['usdc','solanaExplorer','compatibleResearchModels','syncResearchFields'].map(extract).join('\n');
const nodes=new Map(),texts={};
function node(id){if(!nodes.has(id))nodes.set(id,{value:'',hidden:false,disabled:false,innerHTML:'',insertAdjacentHTML(position,html){this.innerHTML=html+this.innerHTML;}});return nodes.get(id);}
let modelCatalog={models:[
  {id:'verified-responses',capability_source:'gateway_advertised',eligible:true,protocols:['responses'],capabilities:{vision:true,structured_actions:true,structured_outputs:true}},
  {id:'verified-messages',capability_source:'owner_declared',eligible:true,protocols:['messages'],capabilities:{vision:true,structured_actions:true,tool_calling:true}},
  {id:'text-only',capability_source:'gateway_advertised',eligible:true,protocols:['responses'],capabilities:{vision:false,structured_actions:true,structured_outputs:true}},
  {id:'no-native-schema',capability_source:'gateway_advertised',eligible:true,protocols:['responses'],capabilities:{vision:true,structured_actions:true}},
  {id:'unverified-prefix',capability_source:'unverified',eligible:false,protocols:['responses'],capabilities:{vision:true,structured_actions:true,structured_outputs:true}},
  {id:'wrong-protocol',capability_source:'gateway_advertised',eligible:true,protocols:['chat'],capabilities:{vision:true,structured_actions:true,structured_outputs:true}}
]};
const context={BigInt,mode:'connected',list:value=>Array.isArray(value)?value:[],catalog:()=>modelCatalog,$:node,text:(id,value)=>texts[id]=value,esc:value=>String(value).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')};
vm.createContext(context);vm.runInContext(functions+'\nthis.tests={usdc,solanaExplorer,compatibleResearchModels,syncResearchFields};',context);
const t=context.tests;
assert.equal(t.usdc('12000001'),'12.000001 USDC');
assert.equal(t.usdc('0'),'0 USDC');
assert.equal(t.usdc(null),'Not available');
assert.equal(t.usdc(1),'Not available');
assert.equal(t.usdc('1000000000000000000000000000000'),'1,000,000,000,000,000,000,000,000 USDC');
assert.equal(t.solanaExplorer('tx','https://evil.example','solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp'),'');
assert.equal(t.solanaExplorer('address','1'.repeat(32),'evil'),'');
assert.ok(t.solanaExplorer('address','1'.repeat(32),'solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1').endsWith('?cluster=devnet'));
assert.deepEqual(Array.from(t.compatibleResearchModels('responses'),m=>m.id),['verified-responses']);
assert.deepEqual(Array.from(t.compatibleResearchModels('messages'),m=>m.id),['verified-messages']);
node('setting-provider').value='x402';node('setting-protocol').value='responses';t.syncResearchFields('old-explicit-model');
assert.equal(node('direct-model-label').hidden,true);
assert.equal(node('research-key-label').hidden,true);
assert.equal(node('setting-catalog-model').value,'old-explicit-model');
assert.ok(node('setting-catalog-model').innerHTML.includes('gateway advertised'));
assert.ok(texts['research-catalog-status'].includes('Paid compatibility remains untested'));
assert.ok(node('setting-catalog-model').innerHTML.includes('unavailable in this catalog'));
node('setting-provider').value='anthropic';t.syncResearchFields();
assert.equal(node('direct-model-label').hidden,false);
assert.equal(node('research-key-label').hidden,false);
assert.equal(node('x402-model-fields').hidden,true);
modelCatalog={models:[]};node('setting-provider').value='x402';t.syncResearchFields('');
assert.equal(node('setting-catalog-model').disabled,true);
assert.ok(texts['research-catalog-status'].includes('No sample models'));
const previewContext={window:{},Date,JSON};vm.createContext(previewContext);
vm.runInContext(fs.readFileSync(path.join(root,'site/observatory-preview.js'),'utf8'),previewContext);
const preview=previewContext.window.ObservatoryPreview.create();
const scope=fs.readFileSync(path.join(root,'observatory/research_scope.py'),'utf8');
const objectiveBlock=scope.match(/DEFAULT_OBJECTIVE = \(([\s\S]*?)\n\)/);
assert.ok(objectiveBlock,'Backend default mission is available');
const objective=Array.from(objectiveBlock[1].matchAll(/"([^"\n]*)"/g),match=>match[1]).join('');
assert.equal(preview.mission.objective,objective,'Fresh preview and backend use the same broad consciousness mission');
const backendRoles=fs.readFileSync(path.join(root,'observatory/research.py'),'utf8');
for(const role of preview.agents)assert.ok(backendRoles.includes('"'+role.specialty+'"'),'Preview role follows the current backend brief: '+role.id);
assert.equal(preview.settings.research_provider,'x402');
assert.equal(preview.settings.browser_provider,'local');
assert.equal(preview.funding.wallet_address,null);
assert.equal(preview.funding.receipts[0].transaction,null);
assert.equal(preview.funding.demo,true);
assert.ok(preview.research_catalog.models.every(m=>m.id.startsWith('simulated/')));
const html=fs.readFileSync(path.join(root,'site/observatory.html'),'utf8');
assert.ok(html.includes(objective+'</textarea>'),'Editable mission starts with the broad default');
assert.ok(html.includes('Saved objectives remain authoritative'),'Saved scopes are not silently widened');
for(const asset of ['observatory.css','observatory-motion.js','observatory.js'])assert.ok(html.includes(asset+'?v=20261006-full1'),'Full-motion updates invalidate cached '+asset);
const htmlIds=Array.from(html.matchAll(/\bid="([^"]+)"/g),m=>m[1]);
assert.equal(new Set(htmlIds).size,htmlIds.length,'HTML IDs must be unique');
const authoredIds=new Set([...htmlIds,...Array.from(source.matchAll(/\bid="([^"]+)"/g),m=>m[1])]);
for(const id of new Set(Array.from(source.matchAll(/\$\("([^"]+)"\)/g),m=>m[1])))assert.ok(authoredIds.has(id),'Missing authored element '+id);
assert.ok(html.includes('id="dataset-export"'));
assert.ok(source.includes('Authorization:"Bearer "+ownerToken'));
assert.ok(!/admin\/datasets[^\n]*\?token/.test(source),'Export token must never be in a URL');
assert.ok(!/<input[^>]*(?:wallet[-_ ]?(?:private|key)|mnemonic|signer|seed_phrase)/i.test(html),'No wallet signing fields');
assert.ok(source.includes('["paused","funding_paused","faulted"].includes(state.mission.status)'));
assert.ok(!html.includes('id="view-funding"'));
assert.ok(!html.includes('data-view="funding"'));
assert.ok(!source.includes('request("funding")'));
assert.ok(html.includes('observatory-motion.js'));
assert.ok(source.includes('qualityReviewMarkup(source,canReview)'));
console.log(JSON.stringify({checks:'passed',assertions:checks,sourceOnly:true,noBrowser:true,noProviderCalls:true}));
