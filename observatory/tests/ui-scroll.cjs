/* Exercise actual viewer helpers without opening a browser or contacting providers. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'../..'),source=fs.readFileSync(path.join(root,'site/observatory.js'),'utf8');
function extract(name){
  const sync=source.indexOf('  function '+name+'('),start=sync>=0?sync:source.indexOf('  async function '+name+'(');assert.ok(start>=0,name);
  const next=source.slice(start+1).search(/\n  (?:async )?function \w+\(/);assert.ok(next>=0,name+' boundary');
  return source.slice(start,start+1+next);
}
const nodes=new Map(),labels={};
function node(id){if(!nodes.has(id))nodes.set(id,{hidden:false,style:{},clientWidth:600,clientHeight:425,getBoundingClientRect:()=>({left:0,top:0})});return nodes.get(id);}
const context={Number,Math,JSON,mode:'connected',currentView:'research',selectedAgent:'scholar',frameTelemetry:null,highlightedNoteId:'',
  previewScrollFrame:null,previewScrollKey:'',previewScrollY:0,previewIdleAt:null,previewTimer:null,motionMode:'full',cancelAnimationFrame(){},matchMedia:()=>({matches:true}),
  window:{},document:{hidden:false},connected:true,state:{events:[],mission:{status:'paused'}},
  $:node,text:(id,value)=>labels[id]=value,agent:()=>({id:'scholar',source_id:'preview-butlin',preview_scroll_phase:2,preview_inspection_id:'preview-inspection-7',preview_focus_note_id:'preview-note-7'})};
vm.createContext(context);
vm.runInContext(['viewportGeometry','frameMetadata','resetFrameTelemetry','motionReduced','renderFrameTelemetry','lockObserverViewport','cancelPreviewScroll','previewPassageLines','paintPreviewViewport','renderPreviewViewport','updatePreviewTimer','refreshFrame'].map(extract).join('\n')+
  '\nthis.helpers={viewportGeometry,frameMetadata,resetFrameTelemetry,motionReduced,renderFrameTelemetry,lockObserverViewport,renderPreviewViewport,updatePreviewTimer,refreshFrame};',context);
const h=context.helpers,viewport={viewport_width:1000,viewport_height:800,scroll_x:0,scroll_y:1200,document_width:1000,document_height:4000};
const focus={x:100,y:200,width:200,height:40,kind:'supporting_passage',note_id:'note-7'};
const geometry=h.viewportGeometry(viewport,focus,600,425);
assert.equal(geometry.start,.3);assert.equal(geometry.extent,.2);assert.equal(geometry.end,.5);
assert.equal(geometry.rect.left,87.5);assert.equal(geometry.rect.top,106.25);
assert.equal(geometry.rect.width,106.25);assert.equal(geometry.rect.height,21.25);
// Correct letterboxing for a portrait display; overlays follow the image, not the outer box.
const portrait=h.viewportGeometry(viewport,focus,300,500);
assert.equal(portrait.rect.left,30);assert.equal(portrait.rect.top,190);
for(const value of [NaN,Infinity,-1,true,'4000'])assert.equal(h.viewportGeometry({...viewport,document_height:value},focus,600,425),null);
assert.equal(h.viewportGeometry({...viewport,scroll_y:4000},focus,600,425),null);
assert.equal(h.viewportGeometry({...viewport,document_width:999},focus,600,425),null);
assert.equal(h.viewportGeometry(viewport,focus,0,425),null);
for(const invalid of [{...focus,x:-1},{...focus,width:Infinity},{...focus,y:799,height:2},{...focus,note_id:'<script>'},{...focus,kind:'guessed_reading'}]){
  assert.equal(h.viewportGeometry(viewport,invalid,600,425).rect,null);
}
function headers(overrides={}){const data={'X-Observatory-Agent':'scholar','X-Observatory-Frame-SHA256':'a'.repeat(64),'X-Observatory-Viewport':JSON.stringify(viewport),'X-Observatory-Focus':JSON.stringify(focus),...overrides};return {get:key=>data[key]??null};}
assert.equal(h.frameMetadata(headers(),'scholar').focus.note_id,'note-7');
assert.equal(h.frameMetadata(headers(),'skeptic'),null);
assert.equal(h.frameMetadata(headers({'X-Observatory-Frame-SHA256':null}),'scholar'),null);
assert.equal(h.frameMetadata(headers({'X-Observatory-Viewport':'broken'}),'scholar'),null);
assert.equal(h.frameMetadata(headers({'X-Observatory-Viewport':'x'.repeat(2049)}),'scholar'),null);
assert.equal(h.frameMetadata(headers({'X-Observatory-Focus':JSON.stringify({...focus,kind:'preview_passage'})}),'scholar').focus,null);
const inspection={...focus,kind:'inspection_passage',inspection_id:'inspection-1',lines:[{x:100,y:200,width:200,height:20},{x:100,y:224,width:160,height:20}]};delete inspection.note_id;
assert.equal(h.frameMetadata(headers({'X-Observatory-Focus':JSON.stringify(inspection)}),'scholar').focus,null,'Live inspection needs document identity');
assert.equal(h.frameMetadata(headers({'X-Observatory-Focus':JSON.stringify({...focus,inspection_id:'inspection-1',lines:inspection.lines})}),'scholar').focus,null,'Promoted inspection needs document identity');
const inspected=h.frameMetadata(headers({'X-Observatory-Document':'b'.repeat(64),'X-Observatory-Focus':JSON.stringify(inspection)}),'scholar');
assert.equal(inspected.documentKey,'b'.repeat(64));assert.equal(inspected.focus.inspection_id,'inspection-1');
const lineGeometry=h.viewportGeometry(viewport,inspection,600,425);
assert.equal(lineGeometry.kind,'inspection_passage');assert.equal(lineGeometry.lines.length,2);assert.equal(lineGeometry.lines[1].top,119);
assert.equal(h.viewportGeometry(viewport,{...inspection,lines:[{x:100,y:200,width:Infinity,height:20}]},600,425).rect,null);
assert.equal(h.viewportGeometry(viewport,{...inspection,lines:Array(13).fill(inspection.lines[0])},600,425).rect,null);
assert.equal(h.viewportGeometry(viewport,{...inspection,lines:undefined},600,425).rect,null,'Inspection needs painted lines');
assert.equal(h.viewportGeometry(viewport,{...focus,inspection_id:'inspection-1'},600,425).rect,null,'Promoted inspection needs painted lines');
assert.equal(h.frameMetadata(headers({'X-Observatory-Document':'private-cdp-id'}),'scholar'),null);
context.frameTelemetry={viewport,focus};h.renderFrameTelemetry();
assert.equal(node('passage-focus').hidden,false);assert.equal(context.highlightedNoteId,'note-7');
assert.equal(node('passage-focus').style.left,'87.5px');assert.equal(labels['viewport-position'],'Agent viewport · 30–50% of page');
// A historical note on a revisited source cannot masquerade as a fresh save.
let motionInput;
context.window.ObservatoryMotion={update:value=>motionInput=value,stop(){}};
context.state.events=[{id:1,type:'note.saved',agent_id:'scholar',created_at:new Date(Date.now()-60000).toISOString(),data:{note_id:'note-7'}}];
h.renderFrameTelemetry();assert.equal(motionInput.event,undefined);
const savedEvent={id:2,type:'note.saved',agent_id:'scholar',created_at:new Date().toISOString(),data:{note_id:'note-7',inspection_id:'inspection-1'}};
context.state.events=[savedEvent];context.frameTelemetry={viewport,focus:{...focus,inspection_id:'inspection-1',lines:inspection.lines}};
h.renderFrameTelemetry();assert.equal(motionInput.event.id,2);
context.state.events=[{...savedEvent,data:{...savedEvent.data,inspection_id:'other-selection'}}];h.renderFrameTelemetry();assert.equal(motionInput.event,undefined);
context.state.events=[];
// A missing/mismatched frame never keeps a passage marker from another capture or agent.
h.resetFrameTelemetry();h.renderFrameTelemetry();
assert.equal(node('passage-focus').hidden,true);assert.equal(node('agent-scroll-track').hidden,true);assert.equal(context.highlightedNoteId,'');
const events={};h.lockObserverViewport({addEventListener:(name,handler,options)=>events[name]={handler,options}});
let prevented=0;
for(const name of ['wheel','touchmove','dragstart'])events[name].handler({preventDefault:()=>prevented++});
assert.equal(prevented,3);assert.equal(events.wheel.options.passive,false);assert.equal(events.touchmove.options.passive,false);
// Preview scroll is changed by research steps, never by a visitor manipulating an iframe.
const page={scrollHeight:760,style:{},querySelector:()=>({offsetTop:520,offsetLeft:30,offsetWidth:420,offsetHeight:90})};
node('preview-frame').querySelector=()=>page;
context.document.createRange=()=>({selectNodeContents(){},getClientRects(){const scroll=Number((page.style.transform || '').match(/-(\d+)/)?.[1] || 0);return [0,1,2].map(i=>({left:30,right:450,top:520+i*28-scroll,bottom:540+i*28-scroll}));}});
context.mode='preview';h.renderPreviewViewport();
assert.equal(page.style.transform,'translateY(-335px)');
assert.equal(context.frameTelemetry.viewport.scroll_y,335);assert.equal(context.frameTelemetry.focus.y,185);
assert.equal(context.frameTelemetry.focus.lines.length,3);
assert.equal(context.highlightedNoteId,'preview-note-7');assert.equal(labels['passage-focus-caption'],'Example passage');
assert.equal(node('passage-focus').hidden,true,'Line highlights replace the instant paragraph box');
context.agent=()=>({preview_scroll_phase:2,preview_focus_note_id:null,preview_inspection_id:null});h.renderPreviewViewport();
assert.equal(context.frameTelemetry.focus,null);assert.equal(node('passage-focus').hidden,true);
// The actual preview document moves progressively; highlights wait until it settles.
let scrollFrame=null,scrollCancelled=0;
context.requestAnimationFrame=callback=>{scrollFrame=callback;return 1;};
context.cancelAnimationFrame=()=>{scrollFrame=null;scrollCancelled++;};context.performance={now:()=>0};
context.agent=()=>({id:'scholar',source_id:'preview-butlin',preview_scroll_phase:2,preview_inspection_id:'preview-inspection-7',preview_focus_note_id:'preview-note-7'});
context.previewScrollY=0;context.previewScrollKey='';context.state.mission.status='running';
h.renderPreviewViewport();assert.equal(typeof scrollFrame,'function');
scrollFrame(325);assert.equal(page.style.transform,'translateY(-168px)');assert.equal(context.frameTelemetry.focus,null);
assert.equal(motionInput.scrolling,true,'The controller receives explicit scroll continuity');
scrollFrame(650);assert.equal(context.previewScrollFrame,null);assert.equal(context.frameTelemetry.focus.lines.length,3);
assert.equal(motionInput.scrolling,false);
context.previewScrollY=0;h.renderPreviewViewport();context.state.mission.status='paused';scrollFrame(325);
assert.equal(scrollFrame,null);assert.ok(scrollCancelled>0,'Pausing cancels the scroll rather than completing it in the background');
context.state.mission.status='running';context.previewScrollY=0;context.previewScrollKey='';h.renderPreviewViewport();scrollFrame(325);
const pausedTransform=page.style.transform;
context.state.mission.status='paused';h.renderPreviewViewport();
assert.equal(page.style.transform,pausedTransform,'Rendering pause freezes the document midway instead of jumping to its target');
assert.equal(context.previewScrollFrame,null);assert.equal(motionInput.running,false,'Pause is delivered to the motion controller immediately');
// Full app playback stays animated even when the OS requests reduced motion.
assert.equal(h.motionReduced(),false,'The default keeps the progressive scroll checked above active on reduced-motion systems');
context.matchMedia=()=>({matches:false});assert.equal(h.motionReduced(),false,'An OS preference change does not change the app playback mode');
// The real preview scheduler cannot replace a passage halfway through its timeline.
let timerCallback=null,timerClock=0,controllerBusy=true,steps=0,renders=0;
Object.assign(context,{previewScrollFrame:null,previewTimer:null,previewIdleAt:null,currentView:'research',
  setTimeout:callback=>{timerCallback=callback;return 9;},clearTimeout:()=>{timerCallback=null;},
  performance:{now:()=>timerClock},render:()=>renders++,
  window:{ObservatoryMotion:{isBusy:()=>controllerBusy},ObservatoryPreview:{step:()=>steps++}}});
context.state.mission.status='running';h.updatePreviewTimer();
timerClock=7000;timerCallback();assert.equal(steps,0,'A long actual passage is never interrupted by a six-second timer');
controllerBusy=false;timerCallback();assert.equal(steps,0,'Completed passage gets a brief hold');
timerClock+=450;timerCallback();assert.equal(steps,1);assert.equal(renders,1);
context.previewScrollFrame=12;timerClock+=5000;timerCallback();assert.equal(steps,1,'Scroll must finish before the next passage');
context.previewScrollFrame=null;context.document.hidden=true;timerCallback();assert.equal(steps,1,'Hidden preview cannot advance');
context.document.hidden=false;context.currentView='evidence';timerCallback();assert.equal(steps,1,'Other views do not skip unseen passages');
context.currentView='research';context.state.mission.status='paused';h.updatePreviewTimer();assert.equal(timerCallback,null,'Pause clears the scheduled callback');
const previewContext={window:{},Date,JSON};vm.createContext(previewContext);
vm.runInContext(fs.readFileSync(path.join(root,'site/observatory-preview.js'),'utf8'),previewContext);
const preview=previewContext.window.ObservatoryPreview,state=preview.create(),original=state.agents[0].source_id;
for(let phase=0;phase<3;phase++){preview.step(state,'scholar');assert.equal(state.agents[0].preview_scroll_phase,phase);assert.equal(state.agents[0].source_id,original);}
assert.ok(state.agents[0].preview_focus_note_id);assert.ok(state.notes.some(note=>note.id===state.agents[0].preview_focus_note_id));
assert.equal(state.notes.at(-1).support_verified,false,'Preview passage is never certified as original evidence');
assert.equal(state.notes.at(-1).inspection_id,state.agents[0].preview_inspection_id);
assert.equal(state.notes.at(-1).passage,state.sources[0].limitation);
const html=fs.readFileSync(path.join(root,'site/observatory.html'),'utf8'),css=fs.readFileSync(path.join(root,'site/observatory.css'),'utf8');
assert.ok(!/<iframe\b/i.test(html));assert.ok(!source.includes('.srcdoc'));
assert.ok(html.includes('Agent controls scrolling'));assert.ok(css.includes('.preview-viewport { position: absolute; inset: 0; overflow: hidden; pointer-events: none;'));
async function checkRejectedOldFrames(){
  for(const nextMode of ['connected','preview']){
    let release,mutations=0;
    Object.assign(context,{mode:'connected',connected:true,currentView:'research',document:{hidden:false},frameLoading:false,frameSelection:'a',frameGeneration:1,selectedAgent:'a',agent:()=>({id:'a'}),
      endpoint:value=>value,fetch:()=>new Promise(resolve=>release=resolve),AbortController,setTimeout,clearTimeout,
      resetFrameTelemetry:()=>mutations++,renderNotebook:()=>mutations++,text:()=>mutations++});
    const pending=h.refreshFrame();
    context.mode=nextMode;context.selectedAgent='b';context.frameGeneration=2;context.agent=()=>({id:'b'});
    release({ok:false,status:404});await pending;
    assert.equal(mutations,0,'A late failed frame cannot change another agent or preview');
  }
}
checkRejectedOldFrames().then(()=>console.log(JSON.stringify({viewerChecks:'passed',noBrowser:true,noProviderCalls:true}))).catch(error=>{console.error(error);process.exitCode=1;});
