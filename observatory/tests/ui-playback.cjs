/* Coupled, deterministic playback of the real viewer, motion and preview code.
 * DOM text rectangles are authored layout fixtures. No browser, HTTP, AI call,
 * screenshot, payment or training job is started by this integration regression. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'../..');
const main=fs.readFileSync(path.join(root,'site/observatory.js'),'utf8');
const motion=fs.readFileSync(path.join(root,'site/observatory-motion.js'),'utf8');
const preview=fs.readFileSync(path.join(root,'site/observatory-preview.js'),'utf8');
const defaultMotionMode=vm.runInNewContext(main.match(/^  const motionMode=.*;$/m)[0]+'\nmotionMode;');
let assertions=0;
const check=(condition,message)=>{assert.ok(condition,message);assertions++;};
const equal=(actual,expected,message)=>{assert.equal(actual,expected,message);assertions++;};
function extract(name) {
  const sync=main.indexOf('  function '+name+'('),start=sync>=0?sync:main.indexOf('  async function '+name+'(');
  assert.ok(start>=0,'Real helper '+name+' exists');
  const next=main.slice(start+1).search(/\n  (?:async )?function \w+\(/);
  assert.ok(next>=0,'Real helper '+name+' has a bounded source block');
  return main.slice(start,start+1+next);
}

function harness({systemReduced=true,summaryLines=5}={}) {
  const nodes=new Map(),rafs=new Map(),timeouts=new Map(),samples=[],renders=[];
  let clock=0,sequence=0,context;
  function element(id='') {
    const classes=new Set(),attributes={};let html='';
    const node={id,style:{},dataset:{},hidden:false,disabled:false,children:[],parentElement:null,textContent:'',clientWidth:600,clientHeight:425,
      rect:{left:100,top:80,width:600,height:425},
      classList:{add:(...items)=>items.forEach(item=>classes.add(item)),remove:(...items)=>items.forEach(item=>classes.delete(item)),
        toggle:(item,on)=>on?classes.add(item):classes.delete(item),contains:item=>classes.has(item)},
      setAttribute(name,value){attributes[name]=String(value);},
      replaceChildren(...children){this.children=children;children.forEach(child=>child.parentElement=this);},
      getBoundingClientRect(){return {...this.rect,right:this.rect.left+this.rect.width,bottom:this.rect.top+this.rect.height};},
      closest(selector){let item=this;while(item){if(selector==='[hidden]'&&item.hidden)return item;item=item.parentElement;}return null;},
      querySelector(selector){return this.subnodes?.[selector] || null;}}
    Object.defineProperty(node,'innerHTML',{get:()=>html,set:value=>{
      html=String(value);
      if(id==='preview-frame') {
        if(!html){node.subnodes={};return;}
        const page=element();page.scrollHeight=1680;page.sections=new Map();
        // The actual illustrativeArticle output determines text and section IDs;
        // these line shapes emulate a wrapped viewport without interpreting HTML.
        const matches=html.matchAll(/data-preview-section="(\d+)">([\s\S]*?)<\/(?:p|h1)>/g);
        for(const match of matches) {
          const section=Number(match[1]),target=element();target.section=section;
          target.textContent=match[2].replace(/<[^>]*>/g,'');
          target.offsetTop=[110,360,790,1230][section];
          target.lineCount=section===1?summaryLines:section===0?2:4;target.parentElement=page;
          page.sections.set(section,target);
        }
        page.querySelector=selector=>page.sections.get(Number(selector.match(/data-preview-section="(\d+)"/)?.[1])) || null;
        node.subnodes={'.preview-page':page};
      }
    }});
    if(id)nodes.set(id,node);return node;
  }
  function $(id){return nodes.get(id) || element(id);}
  const display=$('browser-display'),marker=$('selected-creature'),layout=$('research-layout'),receipt=$('notebook-capture');
  marker.subnodes={'.saw-blade':element(),'.saw-body':element()};
  layout.rect={left:20,top:30,width:950,height:650};
  receipt.rect={left:745,top:185,width:220,height:110};receipt.hidden=true;receipt.parentElement=$('notebook');
  $('evidence-packet').hidden=true;$('evidence-seal').hidden=true;
  $('note-form').subnodes={button:element()};
  const tabs=['decisions','notes','extracts'].map(name=>{const node=element();node.dataset.notebook=name;return node;});
  const media={matches:systemReduced,listeners:[],addEventListener(type,callback){if(type==='change')this.listeners.push(callback);}};
  const document={hidden:false,getElementById:$,createElement:()=>element(),
    querySelectorAll(selector){
      if(selector==='[data-notebook]')return tabs;
      const identifier=selector.match(/^\[data-note-id="([A-Za-z0-9_-]+)"\]$/)?.[1];
      return identifier?[...nodes.values()].filter(node=>node.dataset.noteId===identifier):[];
    },
    createRange(){let target;return {selectNodeContents(value){target=value;},getClientRects(){
      const page=target.parentElement,scroll=Number((page.style.transform || '').match(/translateY\(-([\d.]+)px\)/)?.[1] || 0);
      return Array.from({length:target.lineCount},(_,index)=>{
        const left=display.rect.left+40,top=display.rect.top+target.offsetTop-scroll+index*26;
        const width=index===target.lineCount-1?250:420;
        return {left,top,right:left+width,bottom:top+20,width,height:20};
      });
    }};}}
  context={window:{},document,Math,String,Number,Array,Set,JSON,Date,URL,
    performance:{now:()=>clock},matchMedia:()=>media,
    requestAnimationFrame:callback=>{const id=++sequence;rafs.set(id,callback);return id;},
    cancelAnimationFrame:id=>rafs.delete(id),
    setTimeout:(callback,delay=0)=>{const id=++sequence;timeouts.set(id,{callback,due:clock+delay});return id;},
    clearTimeout:id=>timeouts.delete(id),
    getComputedStyle:node=>({display:'block',visibility:'visible',overflow:'visible',overflowX:'visible',overflowY:'visible',...node.computed}),
    $:$,all:selector=>document.querySelectorAll(selector),list:value=>Array.isArray(value)?value:[],
    esc:value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char])),
    mode:'preview',currentView:'research',selectedAgent:'scholar',notebook:'decisions',motionMode:defaultMotionMode,
    connected:false,ownerToken:'',frameObjectUrl:'',frameSelection:'',lastPreviewSource:'',
    frameTelemetry:null,highlightedNoteId:'',previewScrollFrame:null,previewScrollKey:'',previewScrollY:0,
    previewIdleAt:null,previewTimer:null,sourceBookmarks:new Set(),toast:()=>{}};
  vm.createContext(context);vm.runInContext(preview,context);vm.runInContext(motion,context);
  context.state=context.window.ObservatoryPreview.create();
  const names=['text','titleCase','clock','short','sourceUrl','agentName','sourceById','agent','currentSource','canMutate',
    'viewportGeometry','motionReduced','resetFrameTelemetry','renderFrameTelemetry','illustrativeArticle','cancelPreviewScroll',
    'previewPassageLines','paintPreviewViewport','renderPreviewViewport','creature','renderBrowser','renderNotebook','updatePreviewTimer','missionAction'];
  vm.runInContext(names.map(extract).join('\n'),context);
  context.render=()=>{
    renders.push({time:clock,step:context.agent().step,busy:context.window.ObservatoryMotion.isBusy()});
    context.renderNotebook();context.renderBrowser();context.renderNotebook();
  };
  context.render();
  function snapshot() {
    const selected=context.agent();
    return {time:clock,phase:selected.preview_scroll_phase??-1,step:selected.step,
      running:context.state.mission.status==='running',scroll:context.previewScrollY,
      scrolling:context.frameTelemetry?.scrolling===true,busy:context.window.ObservatoryMotion.isBusy(),
      x:parseFloat(marker.style.left),y:parseFloat(marker.style.top),hidden:marker.hidden,action:marker.dataset.action,
      widths:$('passage-trail').children.map(node=>parseFloat(node.style.width) || 0),
      packet:!$('evidence-packet').hidden,packetTransform:$('evidence-packet').style.transform,
      noteId:receipt.dataset.noteId,inspectionId:context.frameTelemetry?.focus?.inspection_id};
  }
  function frame() {
    clock+=16;
    for(const [id,item] of [...timeouts].sort((a,b)=>a[1].due-b[1].due)) {
      if(item.due<=clock && timeouts.has(id)){timeouts.delete(id);item.callback();}
    }
    for(const [id,callback] of [...rafs]) {
      if(rafs.has(id)){rafs.delete(id);callback(clock);}
    }
    const sample=snapshot();samples.push(sample);return sample;
  }
  function advance(ms){const end=clock+ms;while(clock<end)frame();return snapshot();}
  function until(predicate,maxMs=30000) {
    const deadline=clock+maxMs;
    while(!predicate(snapshot())){assert.ok(clock<deadline,'Playback condition reached before timeout');frame();}
    return snapshot();
  }
  return {context,nodes,rafs,timeouts,samples,renders,media,snapshot,frame,advance,until,
    start(){context.missionAction('start');return snapshot();},pause(){context.missionAction('pause');return snapshot();},
    resume(){context.missionAction('resume');return snapshot();},stop(){context.missionAction('stop');},
    node:$};
}

// Full playback must stay animated despite the OS reporting reduced motion.
// Actual timer + actual viewer + actual controller drive all four fixture phases.
{
  const h=harness();h.start();
  equal(h.media.matches,true,'This regression reproduces the reduced OS environment');
  equal(h.context.motionReduced(),false,'Default full playback continues under an OS reduced-motion preference');
  check(h.snapshot().busy,'The real controller reports the initial inspection as busy');
  h.until(sample=>sample.phase===3 && !sample.busy && !sample.scrolling);
  const phases=new Set(h.samples.map(sample=>sample.phase));
  for(const phase of [0,1,2,3])check(phases.has(phase),'Actual preview phase '+phase+' was displayed');
  const inspect=h.samples.filter(sample=>sample.phase===0 && sample.action==='inspect');
  const partial=inspect.filter(sample=>sample.widths[0]>0 && sample.widths[0]<420);
  check(partial.length>20,'60fps samples observe gradual line reveal, not a completed highlight jump');
  check(new Set(partial.map(sample=>sample.x.toFixed(2))).size>20,'Saw has many distinct intermediate positions on an actual line');
  for(let index=1;index<partial.length;index++) {
    check(Math.abs(partial[index].x-partial[index-1].x)<30,'Adjacent 16ms line samples move smoothly');
    check(partial[index].widths[0]>=partial[index-1].widths[0],'Highlight advances monotonically within the first line');
  }
  const scroll=h.samples.filter(sample=>sample.phase===1 && sample.scrolling);
  check(scroll.length>20,'Page scroll has intermediary 16ms frames');
  check(scroll.every(sample=>!sample.hidden),'The saw remains visible during the authored scroll');
  check(new Set(scroll.map(sample=>sample.scroll)).size>15,'The document actually changes position gradually');
  check(scroll.every(sample=>sample.widths.length===0),'Obsolete text highlights are cleared while the document moves');
  check(new Set(scroll.map(sample=>sample.x.toFixed(2))).size===1,'Scroll carries the current horizontal pose without gutter reset');
  const save=h.samples.filter(sample=>sample.phase===2 && sample.packet);
  check(save.length>20,'A matching saved note has a visible notebook delivery sequence');
  check(new Set(save.map(sample=>sample.packetTransform)).size>20,'Delivery travels through intermediary positions');
  const note=h.context.state.notes.at(-1),event=h.context.state.events.find(item=>item.type==='note.saved' && item.data.note_id===note.id);
  equal(event.data.inspection_id,note.inspection_id,'Real fixture save remains attached to the same inspection');
  check(save.every(sample=>sample.noteId===note.id),'Packet recipient is the actual selected note receipt');
  equal(note.generated_by,'preview_fixture','Playback remains explicitly authored preview content');
  equal(note.support_verified,false,'A preview does not manufacture real provenance verification');
  const transitions=h.renders.filter((item,index,array)=>index>0 && item.step!==array[index-1].step);
  check(transitions.every(item=>!item.busy),'The completion-aware timer never advances an active character timeline');
  equal(h.context.agent().step,4,'Summary, caveat, save and provenance finish in order');
  h.stop();
}

// A long, narrowly wrapped section takes over six seconds, so a fixed cadence
// would interrupt it. The actual completion-aware timer must wait for the path.
{
  const h=harness({summaryLines:12});h.start();h.advance(6500);
  equal(h.context.agent().step,1,'The old six-second boundary cannot skip the long summary');
  equal(h.context.agent().preview_scroll_phase,0,'The inspected section remains selected');
  check(h.snapshot().busy,'The actual twelve-line traversal is still in progress');
  check(h.snapshot().widths.some(width=>width>0) && h.snapshot().widths.some(width=>width===0),
    'At the old boundary the long passage has revealed lines and unread lines, even during a curved return');
  h.until(sample=>sample.phase===1);
  check(h.snapshot().time>8500,'The next phase waits for the complete passage and endpoint hold');
  h.stop();
}

// Pause freezes the partial pose and trail; resume preserves the preview phase.
{
  const h=harness();h.start();h.advance(1400);
  const before=h.snapshot();check(before.busy && before.widths[0]>0,'Pause occurs during an actual partial inspection');
  h.pause();const paused=h.snapshot();h.advance(1000);
  equal(h.snapshot().x,paused.x,'Paused saw position stays fixed');
  equal(h.snapshot().y,paused.y,'Paused saw vertical position stays fixed');
  equal(JSON.stringify(h.snapshot().widths),JSON.stringify(paused.widths),'Paused trail does not finish automatically');
  equal(h.snapshot().scroll,paused.scroll,'Paused document stays fixed');
  equal(h.snapshot().step,before.step,'No timer advances a paused mission');
  equal(h.context.window.ObservatoryMotion.isBusy(),false,'Pause releases the controller busy gate');
  h.resume();equal(h.snapshot().step,before.step,'Resume does not request another fixture step');
  equal(h.snapshot().phase,before.phase,'Resume retains the same selected section');
  check(h.snapshot().busy,'Full motion restarts the current inspection after resume');
  h.stop();
}

// Pause during the page tween must cancel RAF before the same-key early return.
{
  const h=harness();h.start();h.until(sample=>sample.phase===1 && sample.scrolling);h.advance(200);
  check(h.snapshot().scrolling,'Pause is sampled inside the 650ms page tween');
  const before=h.snapshot();h.pause();const transform=h.node('preview-frame').querySelector('.preview-page').style.transform;
  h.advance(1000);
  equal(h.node('preview-frame').querySelector('.preview-page').style.transform,transform,'The document freezes at its intermediate position');
  equal(h.snapshot().scroll,before.scroll,'Pause does not snap to the destination');
  equal(h.rafs.size,0,'Neither character nor document keeps a paused RAF');
  h.resume();equal(h.snapshot().phase,before.phase,'Resuming the scroll does not skip the caveat');
  h.until(sample=>sample.phase===1 && !sample.scrolling && sample.busy);
  check(h.snapshot().scroll>before.scroll,'The resumed page tween reaches the same section');
  h.stop();
}

// Full is the actual app default. OS preference changes cannot interrupt or
// replace the moving inspection with a static completed passage.
{
  const h=harness();h.start();
  h.until(sample=>sample.busy && sample.action==='inspect' && sample.widths[0]>0 && sample.widths[0]<420,6000);
  const before=h.snapshot();
  equal(h.context.motionMode,'full','The viewer uses the actual full app default');
  check(before.busy && before.widths[0]>0 && before.widths[0]<420,'Default playback progressively reveals the passage');
  for(const reduced of [false,true]) {
    h.media.matches=reduced;h.media.listeners.forEach(callback=>callback());
    equal(h.snapshot().busy,true,'OS preference changes keep the active path moving');
    equal(h.snapshot().x,before.x,'Preference changes retain the current character position');
    equal(h.snapshot().widths[0],before.widths[0],'Preference changes retain highlight progress');
  }
  h.advance(160);check(h.snapshot().x!==before.x,'The default character continues through intermediate positions');
  h.stop();
}

process.stdout.write(JSON.stringify({playbackIntegration:'passed',assertions,sampledFrameMs:16,
  realViewerAndController:true,actualAuthoredFixtures:true,allFourPhases:true,fullOverridesReducedOS:true,
  completionAwareTimer:true,pauseAndResume:true,matchingNoteDelivery:true,noNetwork:true})+'\n');
