/* Run the actual motion controller with deterministic DOM/RAF, without a browser. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const source=fs.readFileSync(path.resolve(__dirname,'../../site/observatory-motion.js'),'utf8');
let assertions=0;
const check=(condition,message)=>{assert.ok(condition,message);assertions++;};
const equal=(actual,expected,message)=>{assert.equal(actual,expected,message);assertions++;};

function harness() {
  const nodes=new Map(),frames=new Map();let clock=0,sequence=0;
  function element(id='') {
    const classes=new Set();
    const el={id,style:{},dataset:{},hidden:false,children:[],parentElement:null,textContent:'',clientWidth:600,clientHeight:425,
      classList:{add:(...items)=>items.forEach(x=>classes.add(x)),remove:(...items)=>items.forEach(x=>classes.delete(x)),toggle:(item,on)=>on?classes.add(item):classes.delete(item),contains:item=>classes.has(item)},
      rect:{left:100,top:80,width:600,height:425},
      replaceChildren(...children){this.children=children;children.forEach(child=>child.parentElement=this);},
      getBoundingClientRect(){return {...this.rect,right:this.rect.left+this.rect.width,bottom:this.rect.top+this.rect.height};},
      closest(selector){let node=this;while(node){if(selector==='[hidden]'&&node.hidden)return node;node=node.parentElement;}return null;},
      querySelector(selector){return this.subnodes?.[selector] || null;}};
    if(id)nodes.set(id,el);return el;
  }
  for(const id of ['browser-display','selected-creature','crawler-activity','passage-trail','evidence-seal','evidence-packet','research-layout','notebook-capture'])element(id);
  nodes.get('selected-creature').subnodes={'.saw-blade':element(),'.saw-body':element()};
  nodes.get('research-layout').rect={left:20,top:30,width:950,height:650};
  nodes.get('notebook-capture').rect={left:745,top:185,width:220,height:110};
  nodes.get('notebook-capture').hidden=true;
  nodes.get('evidence-packet').hidden=true;nodes.get('evidence-seal').hidden=true;
  const media={matches:false,listeners:[],addEventListener(type,listener){if(type==='change')this.listeners.push(listener);}};
  const document={hidden:false,getElementById:id=>nodes.get(id)||null,createElement:()=>element(),querySelectorAll:selector=>{
    const id=selector.match(/^\[data-note-id="([A-Za-z0-9_-]+)"\]$/)?.[1];
    return id?[...nodes.values()].filter(node=>node.dataset.noteId===id):[];
  }};
  const context={window:{},document,Math,String,Number,Set,Array,JSON,Date,performance:{now:()=>clock},matchMedia:()=>media,
    requestAnimationFrame:callback=>{const id=++sequence;frames.set(id,callback);return id;},cancelAnimationFrame:id=>frames.delete(id),
    getComputedStyle:node=>({display:'block',visibility:'visible',overflow:'visible',overflowX:'visible',overflowY:'visible',...node.computed})};
  vm.createContext(context);vm.runInContext(source,context);
  return {motion:context.window.ObservatoryMotion,nodes,document,frames,media,element,
    node:id=>nodes.get(id),clock:()=>clock,
    advance(ms){clock=ms;const pending=[...frames.values()];frames.clear();pending.forEach(callback=>callback(ms));},
    setReduced(value){media.matches=value;media.listeners.forEach(callback=>callback({matches:value}));},
    capture(noteId){const node=nodes.get('notebook-capture');node.hidden=false;node.dataset.noteId=noteId;return node;}};
}
function input(overrides={}) {
  return {geometry:{rect:{left:40,top:100,width:300,height:45,inspection_id:'inspect-1',kind:'inspection_passage'},lines:[{left:40,top:100,width:300,height:18},{left:40,top:128,width:300,height:18}],focus_id:'inspect-1',kind:'inspection_passage'},
    viewport:{scroll_y:0,document_height:1600,viewport_height:800},selected:{id:'scholar',current_url:'https://example.org/paper',source_id:'source-1'},
    event:{id:1,type:'agent.section_inspected',agent_id:'scholar',data:{inspection_id:'inspect-1'}},frameKey:'frame-1',documentKey:'document-1',preview:false,running:true,visible:true,motionMode:'system',...overrides};
}
function saved(base=input(),overrides={}) {
  return {...base,geometry:{...base.geometry,rect:{...base.geometry.rect,note_id:'note-1',kind:'supporting_passage'},kind:'supporting_passage',focus_id:'note-1'},
    event:{id:2,type:'note.saved',agent_id:'scholar',data:{note_id:'note-1',source_id:'source-1'}},...overrides};
}
const width=node=>parseFloat(node.style.width);

// Explicit section paths traverse real lines and progressively reveal marks.
{
  const h=harness(),base=input();h.motion.update(base);
  equal(h.frames.size,1,'A selected section starts one finite timeline');
  equal(h.motion.isBusy(),true);
  equal(h.node('selected-creature').dataset.action,'approach');
  equal(h.node('passage-trail').children.length,2);
  check(h.node('passage-trail').children.every(mark=>mark.hidden),'Approach does not prematurely highlight unread lines');
  h.advance(600);equal(h.node('selected-creature').dataset.action,'inspect');
  equal(h.node('selected-creature').style.left,'40px');
  h.advance(1400);
  const marks=h.node('passage-trail').children;
  check(width(marks[0])>145&&width(marks[0])<160,'First line gradually reveals behind the saw');
  equal(marks[1].hidden,true,'The next line waits for the actual traversal');
  check(h.node('selected-creature').querySelector('.saw-blade').style.transform.includes('rotate('));
  check(h.node('selected-creature').querySelector('.saw-body').style.transform!=='rotate(0deg)','Upright hub has a bounded mechanical tilt');
  const firstWidth=width(marks[0]);
  h.motion.update({...base,frameKey:'new-image-2'});
  equal(h.frames.size,1,'New screenshot of identical inspection must not add a competing RAF');
  equal(h.node('passage-trail').children[0],marks[0],'Identical layouts retain their trail nodes');
  h.advance(1900);check(width(marks[0])>firstWidth+80,'Screenshot refresh preserves progress instead of restarting approach');
  h.advance(2300);
  equal(width(marks[0]),300,'Line return keeps the prior line fully marked');
  const turnX=parseFloat(h.node('selected-creature').style.left);
  check(turnX>40&&turnX<340,'Return arcs between the old line end and next line start');
  check(parseFloat(h.node('selected-creature').style.top)>123,'Turn has an actual vertical curve');
  h.advance(2600);check(width(marks[1])>0,'Second line begins after the curved return');
  h.advance(4100);equal(h.motion.isBusy(),true,'Endpoint settling is part of the finite inspection');
  h.advance(4300);equal(h.frames.size,0,'Inspection stops after its finite path');equal(h.motion.isBusy(),false);
  equal(h.node('selected-creature').dataset.action,'idle');
  check(marks.every(mark=>width(mark)===300));
  equal(h.node('evidence-packet').hidden,true,'Inspection alone never claims a notebook save');
  equal(h.node('evidence-seal').hidden,true);
  const left=h.node('selected-creature').style.left;
  h.motion.update({...base,frameKey:'new-image-3',event:{...base.event,id:3}});
  equal(h.frames.size,0,'A generic event cannot restart a completed inspection');
  equal(h.node('selected-creature').style.left,left);
}

// A genuine matching save promotes the already inspected passage; packet targets the note.
{
  const h=harness(),base=input();h.motion.update(base);h.advance(4300);
  const marks=h.node('passage-trail').children;h.capture('note-1');
  h.motion.update(saved(base));
  equal(h.node('passage-trail').children[0],marks[0],'Saving the same passage does not retrace it');
  equal(h.frames.size,1);equal(h.motion.isBusy(),true);h.advance(4301);
  equal(h.node('evidence-seal').hidden,false);equal(h.node('evidence-packet').hidden,false);
  equal(h.node('evidence-packet').textContent,'EVIDENCE','Motion does not expose private source text');
  equal(h.node('selected-creature').dataset.action,'save');
  const start=h.node('evidence-packet').style.transform;h.advance(4750);
  check(h.node('evidence-packet').style.transform!==start,'Packet moves toward the actual note element');
  check(h.node('evidence-packet').style.transform.includes('translate('));
  h.advance(5200);equal(h.node('evidence-packet').hidden,true);equal(h.frames.size,0);equal(h.motion.isBusy(),false);
  equal(h.node('crawler-activity').textContent,'Evidence saved');
  h.motion.update({...saved(base),frameKey:'save-refresh'});equal(h.frames.size,0,'Same saved event delivers once');
  h.motion.update({...saved(base),frameKey:'revisit',documentKey:'reloaded-document'});h.advance(9600);
  equal(h.node('evidence-packet').hidden,true,'Revisiting another document context cannot replay the same note.saved receipt');
  equal(h.frames.size,0);
}

// Matching save while the saw is still tracing waits for the remaining actual lines.
{
  const h=harness(),base=input();h.motion.update(base);h.advance(1400);h.capture('note-1');
  const mark=h.node('passage-trail').children[0],progress=width(mark);
  h.motion.update(saved(base));
  equal(h.node('passage-trail').children[0],mark);equal(width(mark),progress);
  equal(h.node('evidence-packet').hidden,true);
  h.advance(1900);equal(h.node('evidence-packet').hidden,true);
  h.advance(4300);equal(h.node('evidence-packet').hidden,false);
  h.advance(5200);equal(h.frames.size,0);
}

// A different explicit inspection must reacquire even when its line layout is identical.
{
  const h=harness(),base=input();h.motion.update(base);h.advance(1400);
  const oldMark=h.node('passage-trail').children[0];
  const other=saved(base);other.geometry.rect.inspection_id='inspect-2';
  h.motion.update(other);
  check(h.node('passage-trail').children[0]!==oldMark,'Different inspection IDs cannot silently promote an unrelated path');
  equal(h.node('passage-trail').children[0].hidden,true);
}

// A new genuine save after a completed path/stale reset delivers without retracing.
{
  const h=harness(),base=saved(input(),{event:{id:2,type:'agent.decision'}});h.motion.update(base);h.advance(4300);
  h.motion.stop(true);h.capture('note-1');h.motion.update(saved());
  check(h.node('passage-trail').children.every(mark=>width(mark)===300),'Already completed evidence need not be reread for a fresh save');
  h.advance(4301);equal(h.node('evidence-packet').hidden,false);
  h.advance(5200);equal(h.frames.size,0);
}

// Saved geometry without a matching save event does not manufacture delivery.
for(const mismatch of [
  {event:{id:2,type:'agent.decision',data:{note_id:'note-1'}}},
  {event:{id:2,type:'note.saved',data:{note_id:'other-note'}}},
  {event:{id:2,type:'note.saved',agent_id:'skeptic',data:{note_id:'note-1'}}},
  {event:{id:2,type:'note.saved',data:{note_id:'note-1',source_id:'other-source'}}}
]) {
  const h=harness();h.capture('note-1');h.motion.update(saved(input(),mismatch));h.advance(4300);
  equal(h.frames.size,0,'Unassociated events cannot deliver a packet');
  equal(h.node('evidence-packet').hidden,true);equal(h.node('evidence-seal').hidden,true);
}

// Missing, hidden or clipped note destinations safely suppress the flying token.
for(const hidden of ['missing','hidden','clipped']) {
  const h=harness();
  if(hidden==='missing')h.nodes.delete('notebook-capture');
  if(hidden==='clipped') {
    const node=h.capture('note-1'),parent=h.element();parent.rect={left:745,top:50,width:220,height:30};parent.computed={overflow:'hidden'};node.parentElement=parent;
  }
  h.motion.update(saved());h.advance(4300);
  equal(h.node('evidence-packet').hidden,true,'Packet requires a visible current notebook recipient');
  equal(h.node('evidence-seal').hidden,false,'Verified save remains indicated at its passage');
  h.advance(5200);equal(h.frames.size,0);
}

// No selected DOM geometry means no roaming character, fake scan bar or fabricated reading.
{
  const h=harness(),empty=input({geometry:{rect:null,lines:[]}});h.motion.update(empty);
  equal(h.frames.size,0);equal(h.node('selected-creature').hidden,true);
  equal(h.node('passage-trail').hidden,true);equal(h.node('crawler-activity').textContent,'Page opened');
  h.motion.update({...empty,viewport:{...empty.viewport,scroll_y:400}});
  equal(h.node('crawler-activity').textContent,'Agent scrolling');
  equal(h.frames.size,0);equal(h.node('browser-display').classList.contains('is-scanning'),false);
}

// Strict geometry and identity lifecycle guard against stale paths and invalid frame input.
for(const overrides of [
  {frameKey:''},{visible:false},{geometry:null},
  {geometry:{...input().geometry,kind:'preview_inspection'}},
  {geometry:{...input().geometry,lines:[{left:NaN,top:100,width:100,height:18}]}},
  {geometry:{...input().geometry,lines:[{left:590,top:100,width:100,height:18}]}},
  {geometry:{...input().geometry,lines:Array.from({length:13},()=>({left:40,top:100,width:100,height:18}))}}
]) {
  const h=harness();h.motion.update(input(overrides));
  equal(h.frames.size,0);equal(h.node('selected-creature').hidden,true);
}
for(const change of [
  {documentKey:'new-document-same-url'},
  {tabKey:'new-tab-same-url'},
  {selected:{...input().selected,id:'skeptic'}},
  {selected:{...input().selected,current_url:'https://example.org/other'}},
  {viewport:{...input().viewport,scroll_y:400},geometry:{...input().geometry,lines:[{left:40,top:65,width:300,height:18}]}}
]) {
  const h=harness(),base=input();h.motion.update(base);h.advance(1400);
  const oldCallback=[...h.frames.values()][0],oldMark=h.node('passage-trail').children[0];
  h.motion.update({...base,...change,frameKey:'new-frame'});
  equal(h.frames.size,1,'Changed page geometry owns exactly one replacement trajectory');
  check(h.node('passage-trail').children[0]!==oldMark,'Obsolete highlight geometry is discarded');
  equal(h.node('passage-trail').children[0].hidden,true);
  oldCallback(2300);equal(h.frames.size,1,'A cancelled old RAF cannot operate on the newer session');
}
{
  const h=harness(),base=input();h.motion.update(base);h.advance(1400);
  h.node('browser-display').clientWidth=500;h.motion.update({...base,frameKey:'resized'});
  equal(h.frames.size,1);equal(h.node('passage-trail').children[0].hidden,true,'Resize reacquires line geometry');
  h.motion.stop(true);equal(h.frames.size,0);equal(h.node('selected-creature').hidden,true);
  equal(h.node('passage-trail').children.length,0,'Stale stop clears every old line');
  equal(h.node('evidence-packet').hidden,true);
  h.setReduced(true);equal(h.node('selected-creature').hidden,true,'A preference change cannot resurrect cleared stale geometry');
  equal(h.node('passage-trail').children.length,0);
}

// Paused, hidden, stale and reduced-motion spectators never keep animating.
{
  const h=harness(),base=input();h.motion.update(base);h.advance(1400);
  const pausedPosition={left:h.node('selected-creature').style.left,top:h.node('selected-creature').style.top},pausedWidth=width(h.node('passage-trail').children[0]);
  h.motion.update({...base,running:false});
  equal(h.frames.size,0);equal(h.node('crawler-activity').textContent,'Mission paused');
  equal(h.node('selected-creature').classList.contains('is-working'),false);
  equal(h.node('evidence-packet').hidden,true);
  equal(h.node('selected-creature').style.left,pausedPosition.left,'Pause freezes the character instead of jumping to the passage end');
  equal(h.node('selected-creature').style.top,pausedPosition.top);
  equal(width(h.node('passage-trail').children[0]),pausedWidth,'Pause retains the current valid highlight progress');
  equal(h.motion.isBusy(),false);
  h.motion.update({...base,visible:false});equal(h.node('selected-creature').hidden,true);
  equal(h.node('passage-trail').children.length,0);
}
{
  const h=harness();h.motion.update(input());h.document.hidden=true;h.advance(600);
  equal(h.frames.size,0);equal(h.node('selected-creature').hidden,true,'Visibility loss cancels without waiting for next feed');
}
{
  const h=harness(),base=input({preview:true,geometry:{...input().geometry,kind:'preview_inspection'}});h.media.matches=true;h.motion.update(base);
  equal(h.frames.size,0);check(h.node('passage-trail').children.every(mark=>width(mark)===300));
  equal(h.node('crawler-activity').textContent,'Example · Selected section in view');
  equal(h.node('evidence-packet').hidden,true);
  h.capture('note-1');h.motion.update(saved(base,{preview:true,geometry:{...saved(base).geometry,kind:'preview_passage'}}));
  equal(h.frames.size,0);equal(h.node('evidence-seal').hidden,false);
  equal(h.node('crawler-activity').textContent,'Example · Evidence saved');
  h.setReduced(false);equal(h.frames.size,1,'Full motion can replay the passage traversal after a reduced static view');
  h.advance(4300);equal(h.node('evidence-packet').hidden,true,'Opting into motion must not replay a saved receipt');
}
{
  const h=harness();h.motion.update(input());h.advance(1400);h.setReduced(true);
  equal(h.frames.size,0);check(h.node('passage-trail').children.every(mark=>width(mark)===300));
  h.setReduced(false);equal(h.frames.size,1,'Restoring full motion starts an inspectable finite trajectory');
}
// The controller defaults to full even on a reduced-motion OS. Legacy explicit
// controller modes remain compatible; the viewer exposes no mode selector.
for(const mode of [undefined,'full','reduced','system']) {
  const h=harness();h.media.matches=true;const base=input({motionMode:mode});h.motion.update(base);
  const animated=mode===undefined || mode==='full';
  equal(h.motion.isBusy(),animated,'Default and explicit full controller inputs animate despite the OS preference');
  equal(h.frames.size,animated?1:0);
  if(animated) {
    h.advance(300);const x1=parseFloat(h.node('selected-creature').style.left);
    h.advance(450);const x2=parseFloat(h.node('selected-creature').style.left);
    check(x1!==x2&&x2>40,'Approach renders intermediate positions despite OS reduced preference');
    h.advance(1400);h.setReduced(false);equal(h.motion.isBusy(),true,'OS changes cannot interrupt an explicit full-motion opt-in');
  } else {
    equal(h.node('selected-creature').dataset.action,'idle');
    h.motion.update({...base,motionMode:'full'});equal(h.motion.isBusy(),true,'Switching to Full traverses the same currently selected passage');
    equal(h.node('passage-trail').children[0].hidden,true);
  }
}
{
  const h=harness();h.media.matches=false;h.motion.update(input({motionMode:'reduced'}));
  equal(h.frames.size,0,'Explicit Reduced remains static on a device that allows animation');equal(h.motion.isBusy(),false);
}

// Explicit same-document scrolling carries the character's existing pose, with no old highlights.
{
  const h=harness(),base=input({motionMode:'full'});h.motion.update(base);h.advance(1400);
  const oldX=h.node('selected-creature').style.left,oldY=parseFloat(h.node('selected-creature').style.top);
  const scrolling={...base,geometry:{rect:null,lines:[]},viewport:{...base.viewport,scroll_y:25},scrolling:true,frameKey:'scroll-1'};
  h.motion.update(scrolling);
  equal(h.node('selected-creature').hidden,false,'Character remains visible while this observed document scrolls');
  equal(h.node('selected-creature').style.left,oldX);
  equal(parseFloat(h.node('selected-creature').style.top),oldY-25,'Pose follows actual vertical scroll delta');
  equal(h.node('passage-trail').children.length,0,'No obsolete passage highlighting survives scroll');
  equal(h.node('crawler-activity').textContent,'Agent scrolling');equal(h.node('selected-creature').dataset.action,'scroll');
  equal(h.motion.isBusy(),false,'Scroll is driven by observed viewport positions, not a hidden inspection timeline');equal(h.frames.size,0);
  h.motion.update({...scrolling,viewport:{...base.viewport,scroll_y:50},frameKey:'scroll-2'});
  const carriedY=parseFloat(h.node('selected-creature').style.top);
  equal(carriedY,oldY-50);
  h.motion.update({...scrolling,viewport:{...base.viewport,scroll_y:50},running:false});
  equal(h.node('selected-creature').hidden,false,'Mid-scroll pause preserves this document’s existing character pose');
  equal(parseFloat(h.node('selected-creature').style.top),carriedY);
  equal(h.node('passage-trail').children.length,0);equal(h.motion.isBusy(),false);
  const next=input({motionMode:'full',viewport:{...base.viewport,scroll_y:50},geometry:{rect:{left:40,top:80,width:300,height:45,inspection_id:'inspect-2'},lines:[{left:40,top:80,width:300,height:18},{left:40,top:108,width:300,height:18}],focus_id:'inspect-2',kind:'inspection_passage'},frameKey:'next-section'});
  h.motion.update(next);equal(h.node('selected-creature').style.left,oldX,'New section approach begins at retained pose');
  equal(parseFloat(h.node('selected-creature').style.top),carriedY);equal(h.motion.isBusy(),true);
  h.advance(1700);const approachX=parseFloat(h.node('selected-creature').style.left);
  check(approachX>40&&approachX<parseFloat(oldX),'The saw visibly travels between sections instead of skipping');
  h.motion.update({...scrolling,documentKey:'different-document'});equal(h.node('selected-creature').hidden,true,'No pose is carried into a different document');
  h.motion.update({...scrolling,frameKey:''});equal(h.node('selected-creature').hidden,true,'Explicit scroll cannot revive missing/stale frames');
}
{
  const h=harness(),base=input();h.motion.update(base);h.advance(1400);
  const x=h.node('selected-creature').style.left,y=h.node('selected-creature').style.top;
  h.motion.update({...base,running:false});h.motion.update(base);
  equal(h.node('selected-creature').style.left,x);equal(h.node('selected-creature').style.top,y,'Resume reacquires smoothly from the frozen pose');
  equal(h.motion.isBusy(),true);h.motion.stop(true);equal(h.motion.isBusy(),false);
}
{
  const h=harness();const long=input({geometry:{rect:{left:40,top:20,width:100,height:18,inspection_id:'long-inspection'},lines:Array.from({length:12},(_,i)=>({left:40,top:20+i*28,width:100,height:18})),focus_id:'long-inspection',kind:'inspection_passage'}});
  h.motion.update(long);h.advance(4300);equal(h.motion.isBusy(),true,'A longer selected passage receives enough time rather than skipping its latter lines');
  h.advance(9600);equal(h.motion.isBusy(),false);equal(h.frames.size,0,'Even the maximum line count has a finite endpoint');
}
// Wide-line returns travel visibly over hundreds of milliseconds, rather than skipping.
{
  const h=harness(),base=input();base.geometry.lines=base.geometry.lines.map(line=>({...line,width:500}));
  h.motion.update(base);h.advance(2150);
  const earlyX=parseFloat(h.node('selected-creature').style.left),earlyY=parseFloat(h.node('selected-creature').style.top);
  check(earlyX>300&&earlyX<540,'A long curved return still has an intermediate position after 140ms');
  h.advance(2500);
  const laterX=parseFloat(h.node('selected-creature').style.left);
  check(laterX>40&&laterX<earlyX,'Return distance keeps the same turn moving at a later frame');
  check(earlyY>123,'The visible return keeps its curved vertical arc');
  equal(h.node('passage-trail').children[1].hidden,true,'The next line does not reveal before the long return arrives');
  h.advance(4300);equal(h.motion.isBusy(),false,'Ordinary passages retain their finite 3.4s minimum trace');
}
{
  const h=harness();check(h.motion.saw(0).includes('saw-blade'));check(h.motion.saw(0).includes('saw-body'));
  const first=h.motion.saw(0),second=h.motion.saw(0),ids=Array.from(first.matchAll(/\bid="([^"]+)"/g),item=>item[1]);
  check(ids.every(id=>!second.includes('id="'+id+'"')),'Repeated instances cannot share material IDs');
  check(Array.from(first.matchAll(/url\(#([^)]*)\)/g),item=>item[1]).every(id=>ids.includes(id)),'Every material reference resolves inside its own SVG');
  check(!h.motion.saw('<script>').includes('<script>'),'Caller labels cannot inject SVG markup');
}
console.log(JSON.stringify({motionController:'passed',assertions,actualLineTraversal:true,sameInspectionRefreshContinues:true,verifiedNoteDelivery:true,identityAndRaceCancellation:true,motionModes:true,scrollPoseContinuity:true,completionAwareBusy:true,reducedMotion:true,noNetwork:true}));
