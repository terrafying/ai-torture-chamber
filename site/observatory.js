/* Public observer + explicit preview. Owner credentials never enter browser storage. */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const all = selector => Array.from(document.querySelectorAll(selector));
  const esc = value => String(value == null ? "" : value).replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
  const list = value => Array.isArray(value) ? value : [];
  const topicDomains = [["machine_consciousness","Machine consciousness"],["consciousness_science","Consciousness science"],["philosophy_of_mind","Philosophy of mind"],["metaphysics_reality","Reality & metaphysics"],["religion_contemplation","Religion & contemplation"],["welfare_ethics","Welfare & ethics"]];
  const evidenceKinds = [["empirical","Empirical findings"],["scientific_theory","Scientific theory"],["philosophical_argument","Philosophical argument"],["religious_contemplative","Religious or contemplative interpretation"],["mixed","Several forms of evidence or argument"]];
  const autoCurationPolicyAck = "originals-v2";
  const storageKey = "wirehead.observatory.preferences.v1";
  const bookmarkKey = "wirehead.observatory.bookmarks.v1";
  const defaults = {api_base:"/observatory-api",research_provider:"x402",research_model:"",research_protocol:"responses",reasoning_effort:"high",browser_provider:"local",agent_count:6,auto_curation_enabled:false,auto_curation_policy_ack:"",hf_namespace:"",hf_base_model:"meta-llama/Llama-3.1-70B",hf_base_revision:"349b2ddb53ce8f2849a6c168a81980ab25258dac",training_enabled:false,training_continue_from_previous:true,synthetic_training_approved:false,provider_policy_reference:""};
  function mergeSettings(base,overrides={}) {
    const settings={...base,...overrides};
    // A repository change must not inherit another model's default commit.
    // Explicit owner revisions, including null, retain their meaning.
    if(Object.prototype.hasOwnProperty.call(overrides,"hf_base_model") && !Object.prototype.hasOwnProperty.call(overrides,"hf_base_revision")) {
      settings.hf_base_revision=overrides.hf_base_model===base.hf_base_model ? base.hf_base_revision ?? null : overrides.hf_base_model===defaults.hf_base_model ? defaults.hf_base_revision : null;
    }
    return settings;
  }
  let preferences = {...defaults};
  try { preferences = mergeSettings(defaults,JSON.parse(localStorage.getItem(storageKey) || "{}")); } catch (_) {}
  const parameters = new URLSearchParams(location.search);
  if(parameters.has("api")) preferences.api_base=parameters.get("api");
  let mode = parameters.get("preview")==="1" || parameters.get("mode")==="preview" ? "preview" : "connected";
  let state = mode==="preview" ? window.ObservatoryPreview.create() : emptyState();
  if(mode==="preview") {
    state.settings=mergeSettings(state.settings,preferences);
    if(preferences.objective) state.mission.objective=preferences.objective;
  }
  let selectedAgent = "", selectedDataset = "", selectedJob = "", notebook = "decisions", manifestView = "manifest", currentView = "research";
  let ownerToken = "", connected = false, busy = false, connectionError = "", eventSource = null, previewTimer = null, refreshTimer = null, stateTimer = null;
  let frameObjectUrl = "", frameSelection = "", frameLoading = false, frameGeneration = 0, lastPreviewSource = "", toastTimer = null, confirmation = null, eventOnlyAgent = false;
  let frameTelemetry = null, highlightedNoteId = "";
  let previewScrollFrame = null, previewScrollKey = "", previewScrollY = 0;
  const motionMode="full";
  let previewIdleAt=null;
  let inheritedBaseRevision = false;
  let researchCatalog={status:"unavailable",models:[]}, metadataAt=0, metadataLoading=false, metadataGeneration=0;
  let previousProvider=preferences.research_provider;
  let sourceBookmarks = new Set();
  try { sourceBookmarks = new Set(JSON.parse(localStorage.getItem(bookmarkKey) || "[]")); } catch (_) {}

  function emptyState() {
    return {mission:{status:"stopped",until_stopped:true},agents:[],sources:[],notes:[],datasets:[],jobs:[],checkpoints:[],events:[],curation_reviews:[],curation_runtime:null,connections:[],settings:{},cursor:0};
  }
  function normalizeState(payload) {
    if(!payload || typeof payload!=="object" || !payload.mission || !Array.isArray(payload.agents)) throw new Error("The endpoint did not return an observatory state.");
    return {...emptyState(),...payload,agents:list(payload.agents),sources:list(payload.sources),notes:list(payload.notes),datasets:list(payload.datasets),jobs:list(payload.jobs),checkpoints:list(payload.checkpoints),events:list(payload.events),curation_reviews:list(payload.curation_reviews)};
  }
  function text(id,value) { $(id).textContent=value == null ? "" : String(value); }
  function titleCase(value) { return String(value || "").replace(/[_.-]/g," ").replace(/\b\w/g,letter=>letter.toUpperCase()); }
  function adaptationLabel(value) { return ({qlora:"QLoRA",lora:"LoRA"})[String(value || "qlora").toLowerCase()] || titleCase(value); }
  function clock(value) {
    if(!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});
  }
  function dateLabel(value) {
    if(!value) return "Not recorded";
    const date=new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString([],{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit",hour12:false});
  }
  function short(value,length=22) { const string=String(value || "Not recorded"); return string.length>length ? string.slice(0,length)+"…" : string; }
  function safeUrl(value) {
    try { const url=new URL(value); return /^https?:$/.test(url.protocol) && !url.username && !url.password ? url.href : ""; } catch (_) { return ""; }
  }
  function sourceUrl(source) { return source && (source.canonical_url || source.url) || ""; }
  function agentName(id) { const agent=state.agents.find(item=>String(item.id)===String(id)); return agent ? (agent.name || titleCase(agent.role)).replace(/^The /,"") : id ? "Research agent" : "System"; }
  function sourceById(id) { return state.sources.find(source=>String(source.id)===String(id)); }
  function agent() { return state.agents.find(item=>String(item.id)===String(selectedAgent)) || null; }
  function currentSource() {
    const selected=agent();
    if(!selected) return null;
    return sourceById(selected.source_id) || state.sources.find(source=>sourceUrl(source)===selected.current_url) || [...state.sources].reverse().find(source=>source.agent_id===selected.id) || null;
  }
  function eligible(source) { return !!(source.curation && source.curation.eligible || source.eligibility==="eligible"); }
  function rightsClass(source) { return eligible(source) ? "eligible" : source.rights_status==="reference_only" || source.curation?.status==="reference_only" || /NC|distribution/i.test(source.license || "") ? "reference" : "review"; }
  function rightsLabel(source) { return rightsClass(source)==="eligible" ? "Training eligible" : rightsClass(source)==="reference" ? "Reference only" : "Needs review"; }
  function canMutate() { return mode==="preview" || connected && !!ownerToken; }
  function toast(message) {
    text("toast",message);$("toast").hidden=false;clearTimeout(toastTimer);
    toastTimer=setTimeout(()=>$("toast").hidden=true,5000);
  }
  function explainOwner() { toast("Public watch is read-only. Enter the owner access token in Operator setup to use live controls."); openSetup(); }
  function endpoint(path) { return preferences.api_base.replace(/\/+$/,"")+"/"+path.replace(/^\/+/,""); }
  function validateEndpoint(raw) {
    const value=String(raw || "").trim().replace(/\/+$/,"");
    if(/^\/(?!\/)/.test(value) && !/[?#]/.test(value)) return value;
    const url=new URL(value);
    if(!["http:","https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error("Use an HTTP(S) API prefix without credentials, query parameters or fragments.");
    if(url.protocol==="http:" && !["localhost","127.0.0.1","[::1]"].includes(url.hostname)) throw new Error("Use HTTPS for a remote backend. HTTP is allowed for local development.");
    return url.href.replace(/\/+$/,"");
  }
  async function request(path,method="GET",payload) {
    const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),10000);
    try {
      const headers={Accept:"application/json"};
      if(method!=="GET") {
        if(!ownerToken) throw new Error("Owner access token required for live controls.");
        headers.Authorization="Bearer "+ownerToken;headers["Content-Type"]="application/json";
      }
      const response=await fetch(endpoint(path),{method,headers,body:payload===undefined ? undefined : JSON.stringify(payload),signal:controller.signal,cache:"no-store",credentials:"omit"});
      const type=response.headers.get("Content-Type") || "";
      const result=type.includes("application/json") ? await response.json() : null;
      if(!response.ok) throw new Error(typeof result?.detail==="string" ? result.detail : "Backend returned HTTP "+response.status+".");
      if(!result) throw new Error("The backend returned a non-JSON response. Check the API endpoint prefix.");
      return result;
    } catch(error) { if(error.name==="AbortError") throw new Error("Backend request timed out. Check the service and endpoint prefix."); throw error; }
    finally { clearTimeout(timeout); }
  }
  function clearLiveFrame() {
    frameGeneration++;frameSelection="";frameLoading=false;
    if(frameObjectUrl) URL.revokeObjectURL(frameObjectUrl);
    frameObjectUrl="";$("live-frame").removeAttribute("src");$("live-frame").hidden=true;
    resetFrameTelemetry();
  }
  function viewportGeometry(viewport,focus,displayWidth,displayHeight) {
    const keys=["viewport_width","viewport_height","scroll_x","scroll_y","document_width","document_height"];
    if(!viewport || keys.some(key=>typeof viewport[key]!=="number" || !Number.isFinite(viewport[key]) || viewport[key]<0)) return null;
    const v=viewport;
    if(!v.viewport_width || !v.viewport_height || v.viewport_width>32768 || v.viewport_height>32768 || v.document_width<v.viewport_width || v.document_height<v.viewport_height || v.document_width>10000000 || v.document_height>10000000 || v.scroll_x>v.document_width-v.viewport_width || v.scroll_y>v.document_height-v.viewport_height) return null;
    const scale=Math.min(displayWidth/v.viewport_width,displayHeight/v.viewport_height);
    if(!Number.isFinite(scale) || scale<=0) return null;
    const start=v.scroll_y/v.document_height, extent=v.viewport_height/v.document_height;
    const validId=value=>typeof value==="string" && /^[A-Za-z0-9_-]{1,128}$/.test(value);
    const validRect=value=>value && ["x","y","width","height"].every(key=>typeof value[key]==="number" && Number.isFinite(value[key])) && value.x>=0 && value.y>=0 && value.width>0 && value.height>0 && value.x+value.width<=v.viewport_width && value.y+value.height<=v.viewport_height;
    const scaleRect=value=>({left:(displayWidth-v.viewport_width*scale)/2+value.x*scale,top:(displayHeight-v.viewport_height*scale)/2+value.y*scale,width:value.width*scale,height:value.height*scale});
    let rect=null,lines=[],kind="",focus_id="";
    if(focus && ["supporting_passage","inspection_passage","preview_passage","preview_inspection"].includes(focus.kind) && validRect(focus)) {
      const saved=["supporting_passage","preview_passage"].includes(focus.kind);
      const valid=saved?validId(focus.note_id):validId(focus.inspection_id);
      const requiresLines=!saved || focus.inspection_id!==undefined;
      const validLines=!requiresLines && focus.lines===undefined || Array.isArray(focus.lines) && focus.lines.length>0 && focus.lines.length<=12 && focus.lines.every(validRect);
      if(valid && validLines && (focus.inspection_id===undefined || validId(focus.inspection_id))) {
        rect={...scaleRect(focus),kind:focus.kind,note_id:saved?focus.note_id:"",inspection_id:focus.inspection_id || ""};
        lines=(focus.lines || []).map(scaleRect);kind=focus.kind;focus_id=focus.inspection_id || focus.note_id;
      }
    }
    return {start,extent,end:Math.min(1,start+extent),rect,lines,kind,focus_id};
  }
  function frameMetadata(headers,id) {
    if(headers.get("X-Observatory-Agent")!==id || !/^[a-f0-9]{64}$/.test(headers.get("X-Observatory-Frame-SHA256") || "")) return null;
    try {
      const raw=headers.get("X-Observatory-Viewport"), rawFocus=headers.get("X-Observatory-Focus");
      if(!raw || raw.length>2048 || rawFocus && rawFocus.length>2048) return null;
      const viewport=JSON.parse(raw), focus=rawFocus?JSON.parse(rawFocus):null;
      const context=headers.get("X-Observatory-Document") || "";
      if(context && !/^[a-f0-9]{64}$/.test(context)) return null;
      // A connected feed never accepts the preview's authored highlight type.
      const allowed=(focus?.kind==="supporting_passage" || focus?.kind==="inspection_passage") && (!(focus.kind==="inspection_passage" || focus.inspection_id!==undefined) || !!context);
      return {viewport,focus:allowed?focus:null,frameKey:headers.get("X-Observatory-Frame-SHA256"),documentKey:context};
    } catch (_) { return null; }
  }
  function resetFrameTelemetry() {
    frameTelemetry=null;highlightedNoteId="";
    $("passage-focus").hidden=true;$("agent-scroll-track").hidden=true;
    text("viewport-position","Page position unavailable");
    window.ObservatoryMotion?.stop(true);
  }
  function motionReduced() {
    return false;
  }
  function renderFrameTelemetry() {
    const display=$("browser-display"), geometry=viewportGeometry(frameTelemetry?.viewport,frameTelemetry?.focus,display.clientWidth,display.clientHeight);
    highlightedNoteId="";
    $("passage-focus").hidden=!geometry?.rect || !!geometry.lines.length;$("agent-scroll-track").hidden=!geometry;
    if(!geometry) {text("viewport-position","Page position unavailable");window.ObservatoryMotion?.stop(true);return;}
    const start=Math.floor(geometry.start*100),end=Math.min(100,Math.ceil(geometry.end*100));
    text("viewport-position",(mode==="preview"?"Example viewport":"Agent viewport")+" · "+start+"–"+end+"% of page");
    $("agent-scroll-thumb").style.top=geometry.start*100+"%";
    $("agent-scroll-thumb").style.height=geometry.extent*100+"%";
    if(geometry.rect) {
      const rect=geometry.rect, overlay=$("passage-focus");
      overlay.style.left=rect.left+"px";overlay.style.top=rect.top+"px";overlay.style.width=rect.width+"px";overlay.style.height=rect.height+"px";
      text("passage-focus-caption",mode==="preview"?"Example passage":rect.note_id?"Saved note passage":"Selected for inspection");
      highlightedNoteId=rect.note_id || "";
    }
    const selected=agent(), events=[...state.events].reverse().filter(item=>String(item.agent_id)===String(selected?.id));
    const latest=events.find(item=>{
      const age=Date.now()-new Date(item.created_at).getTime();
      return item.type==="note.saved" && item.data?.note_id===geometry.rect?.note_id && (!geometry.rect?.inspection_id || item.data?.inspection_id===geometry.rect.inspection_id) && (mode==="preview" || Number.isFinite(age) && age>=-5000 && age<=30000);
    }) || events.find(item=>item.type!=="note.saved");
    window.ObservatoryMotion?.update({geometry,viewport:frameTelemetry.viewport,selected,event:latest,preview:mode==="preview",scrolling:frameTelemetry.scrolling===true,motionMode,running:state.mission.status==="running" && (mode==="preview" || connected),visible:currentView==="research" && !document.hidden,frameKey:frameTelemetry.frameKey,documentKey:frameTelemetry.documentKey});
  }
  function lockObserverViewport(display) {
    // A wheel or touch gesture over the spectator pane cannot alter its page.
    const hold=event=>event.preventDefault();
    display.addEventListener("wheel",hold,{passive:false});
    display.addEventListener("touchmove",hold,{passive:false});
    display.addEventListener("dragstart",hold);
  }
  function disconnect() {
    if(eventSource) eventSource.close();
    eventSource=null;clearInterval(stateTimer);clearTimeout(refreshTimer);stateTimer=null;connected=false;
    cancelPreviewScroll(true);
    clearLiveFrame();
    metadataGeneration++;metadataAt=0;metadataLoading=false;researchCatalog={status:"unavailable",models:[]};
  }
  async function refreshState(silent=false) {
    if(mode!=="connected") return;
    try {
      const payload=await request("state");
      if(mode!=="connected") return;
      state=normalizeState(payload);connected=true;connectionError="";
      render();
      await refreshPaymentMetadata();
      if(!eventSource) startEventStream();
      if(!stateTimer) stateTimer=setInterval(()=>refreshState(true),12000);
    } catch(error) {
      connected=false;connectionError=error.message;render();
      if(!silent) text("setup-result",error.message+" No preview records were substituted.");
    }
  }
  async function refreshPaymentMetadata(force=false) {
    if(mode==="preview"){syncResearchFields();return;}
    if(!connected || metadataLoading || !force && Date.now()-metadataAt<30000)return;
    const generation=metadataGeneration;metadataLoading=true;
    try {
      const results=await Promise.allSettled([request("research-models")]);
      if(generation!==metadataGeneration || mode!=="connected")return;
      researchCatalog=results[0].status==="fulfilled"?results[0].value:{status:"unavailable",models:[]};
      metadataAt=Date.now();syncResearchFields();
    } finally {if(generation===metadataGeneration)metadataLoading=false;}
  }
  function catalog() { return mode==="preview" ? state.research_catalog || {status:"simulated",models:[]} : researchCatalog; }
  function compatibleResearchModels(protocol) {
    return list(catalog().models).filter(item=>item.eligible===true && ["owner_declared","gateway_advertised"].includes(item.capability_source) && list(item.protocols).includes(protocol) && item.capabilities?.vision===true && item.capabilities?.structured_actions===true && (protocol==="responses"?item.capabilities?.structured_outputs===true:item.capabilities?.tool_calling===true));
  }
  function syncResearchFields(wantedModel) {
    const paid=$("setting-provider").value==="x402";
    $("x402-model-fields").hidden=!paid;$("direct-model-label").hidden=paid;
    if($("research-key-label"))$("research-key-label").hidden=paid;
    const selector=$("setting-catalog-model"), selected=wantedModel===undefined ? selector.value || $("setting-model").value : wantedModel;
    const models=compatibleResearchModels($("setting-protocol").value);
    selector.innerHTML=models.length ? models.map(item=>'<option value="'+esc(item.id)+'">'+esc(item.name && item.name!==item.id?item.name+" · "+item.id:item.id)+' · '+esc(mode==="preview"?"simulated controls":item.capability_source==="owner_declared"?"owner declared":"gateway advertised")+'</option>').join("") : '<option value="">'+esc(mode==="preview"?"No matching simulated model":"No compatible live model available")+'</option>';
    if(models.some(item=>item.id===selected))selector.value=selected;
    else if(selected && paid) {
      selector.insertAdjacentHTML("afterbegin",'<option value="'+esc(selected)+'" disabled>'+esc(selected)+' · unavailable in this catalog</option>');selector.value=selected;
    }
    selector.disabled=!models.length;
    text("research-catalog-status",mode==="preview"?"Simulated catalog. These selections never call a model or make a payment.":models.length?"Controls are owner declared or gateway advertised. Paid compatibility remains untested; the broker validates each request's quote.":"The payment broker is unavailable or has no declared compatible model. Configure the model capability registry outside this interface. No sample models were substituted.");
  }
  function usdc(amount) {
    if(typeof amount!=="string" || !/^\d+$/.test(amount))return "Not available";
    const value=BigInt(amount), whole=value/1000000n, fraction=String(value%1000000n).padStart(6,"0").replace(/0+$/,"");
    return whole.toLocaleString("en-US")+(fraction?"."+fraction:"")+" USDC";
  }
  function solanaExplorer(kind,value,network) {
    if(typeof value!=="string" || !/^[1-9A-HJ-NP-Za-km-z]{32,90}$/.test(value))return "";
    if(!["solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp","solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1"].includes(network))return "";
    return "https://explorer.solana.com/"+kind+"/"+encodeURIComponent(value)+(network==="solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1"?"?cluster=devnet":"");
  }
  function startEventStream() {
    if(mode!=="connected" || !connected || typeof EventSource==="undefined") return;
    eventSource=new EventSource(endpoint("events")+"?after="+encodeURIComponent(state.cursor || 0));
    eventSource.onmessage=event=>{
      try {
        const record=JSON.parse(event.data);
        if(!state.events.some(item=>String(item.id)===String(record.id))) state.events.push(record);
        state.events=state.events.slice(-100);state.cursor=Math.max(state.cursor || 0,record.seq || 0);
        renderEvents();renderNotebook();renderFrameTelemetry();
        clearTimeout(refreshTimer);refreshTimer=setTimeout(()=>refreshState(true),650);
      } catch (_) { toast("An event could not be read. The next state refresh will resynchronize the record."); }
    };
    eventSource.onerror=()=>{
      text("events-caption","Event stream reconnecting · records retained");
    };
  }
  async function refreshFrame() {
    if(mode!=="connected" || !connected || currentView!=="research" || document.hidden || frameLoading) return;
    const selected=agent();if(!selected) return;
    if(frameSelection!==String(selected.id)) { clearLiveFrame();frameSelection=String(selected.id); }
    const generation=frameGeneration, id=String(selected.id);frameLoading=true;let pendingUrl="";
    const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),8000);
    try {
      const response=await fetch(endpoint("agents/"+encodeURIComponent(id)+"/frame")+"?t="+Date.now(),{cache:"no-store",credentials:"omit",signal:controller.signal});
      if(mode!=="connected" || generation!==frameGeneration || String(selectedAgent)!==id) return;
      if(!response.ok) {
        resetFrameTelemetry();renderNotebook();
        if(response.status===404 && !frameObjectUrl) {$("live-frame").hidden=true;$("browser-empty").hidden=false;}
        else if(!response.ok) text("frame-description",frameObjectUrl?"Last captured frame · new frame unavailable":"Waiting for an actual browser frame");
        return;
      }
      if(!/^image\/(png|jpeg|webp)(;|$)/.test(response.headers.get("Content-Type") || "")) throw new Error("Unexpected browser frame format.");
      const blob=await response.blob();
      if(blob.size>10*1024*1024) throw new Error("Browser frame exceeds the preview size limit.");
      if(mode!=="connected" || generation!==frameGeneration || String(selectedAgent)!==id) return;
      const url=URL.createObjectURL(blob);pendingUrl=url;
      const decoded=new Image();decoded.src=url;await decoded.decode();
      if(mode!=="connected" || generation!==frameGeneration || String(selectedAgent)!==id) return;
      const previous=frameObjectUrl;frameObjectUrl=url;pendingUrl="";
      $("live-frame").src=url;$("live-frame").hidden=false;$("browser-empty").hidden=true;
      const capturedAt=response.headers.get("X-Observatory-Frame-At") || response.headers.get("Last-Modified");
      const capturedTime=capturedAt ? new Date(capturedAt).getTime() : NaN;
      const stale=Number.isFinite(capturedTime) && Date.now()-capturedTime>15000;
      text("frame-description","Actual browser capture · "+(Number.isFinite(capturedTime)?clock(capturedAt):"capture time not reported")+(stale?" · stale capture":"")+" · public view is read-only");
      frameTelemetry=stale?null:frameMetadata(response.headers,id);renderNotebook();renderFrameTelemetry();renderNotebook();
      $("frame-stamp").hidden=false;
      if(previous) URL.revokeObjectURL(previous);
    } catch(error) { if(generation===frameGeneration){resetFrameTelemetry();renderNotebook();text("frame-description",frameObjectUrl?"Last actual capture · browser feed temporarily unavailable":"Waiting for browser worker · no simulated frame");} }
    finally { clearTimeout(timeout);if(pendingUrl)URL.revokeObjectURL(pendingUrl);if(generation===frameGeneration) frameLoading=false; }
  }
  function creature(index=0) { return window.ObservatoryMotion.saw(index); }
  function render() {
    state.settings=state.settings || {};
    if(!state.agents.some(item=>String(item.id)===String(selectedAgent))) selectedAgent=state.agents[0]?.id || "";
    if(!state.datasets.some(item=>String(item.id)===String(selectedDataset))) selectedDataset=state.datasets.at(-1)?.id || "";
    if(!state.jobs.some(item=>String(item.id)===String(selectedJob))) selectedJob=state.jobs.at(-1)?.id || "";
    document.body.classList.toggle("connected",mode==="connected");
    document.body.classList.toggle("is-paused",state.mission.status!=="running" || mode==="connected" && !connected);
    renderMode();renderMission();renderAgents();renderNotebook();renderBrowser();renderNotebook();renderEvents();renderEvidence();renderDatasets();renderTraining();renderCheckpoints();renderConnections();
    if($("notes-dialog").open) renderAllNotes();
  }
  function renderMode() {
    const error=mode==="connected" && !connected;
    $("mode-badge").className="mode-badge"+(mode==="preview" ? "" : error ? " error" : " connected");
    $("mode-notice").className="mode-notice"+(mode==="preview" ? "" : error ? " error" : " connected");
    text("mode-badge",mode==="preview" ? "Preview · simulated" : connected ? "Connected · actual data" : "Connected · unavailable");
    text("mode-description",mode==="preview" ? "Interactive preview. Browsing, records and jobs are simulated. No model is training." : connected ? "Actual backend records and browser captures. Public watch is read-only; owner controls require a token." : (connectionError || "Connecting to the observatory backend…")+" No simulated records are shown.");
    text("connect-open",mode==="preview" ? "Connect a backend" : "Connection setup");
    text("foot-mode",mode==="preview" ? "Concept preview" : connected ? "Connected observatory" : "Backend disconnected");
  }
  function renderMission() {
    const status=state.mission.status || "stopped", running=status==="running", transition=["pausing","stopping"].includes(status);
    text("mission-state",(mode==="preview" ? "Preview " : "")+titleCase(status));
    $("mission-state").classList.toggle("running",running);
    text("mission-title",state.mission.objective || "No research mission has been started.");
    const resumable=["paused","funding_paused","faulted"].includes(status);
    text("mission-detail",mode==="preview" ? "An open research frontier. The simulation continues until you stop it." : status==="funding_paused"?"Research is paused for funding. Check the allowance and reserved requests before explicitly resuming.":status==="faulted"?"The worker stopped after a provider or configuration fault. Review the event record and setup before resuming.":transition ? "The worker will checkpoint and release its browser at a safe action boundary." : "An open research frontier. Agents continue until the owner stops the mission.");
    $("mission-toggle").innerHTML=esc(mode==="preview" ? running?"Pause preview":resumable?"Resume preview":"Run preview" : running?"Pause mission":resumable?"Resume mission":"Start mission")+' <span aria-hidden="true">'+(running?"Ⅱ":"▷")+'</span>';
    $("mission-toggle").disabled=busy || transition || mode==="connected" && !connected;
    $("mission-stop").disabled=busy || transition || !["running","paused","pausing","funding_paused","faulted"].includes(status);
    $("mission-step").hidden=mode!=="preview";$("mission-step").disabled=busy;
    text("metric-sources",state.sources.length);text("metric-eligible",state.sources.filter(eligible).length);
    text("metric-questions",state.mission.open_questions ?? state.notes.filter(note=>["question","lead"].includes(note.type)).length);
    text("metric-label",mode==="preview" ? "Example records" : "Collected records");
  }
  function renderAgents() {
    text("research-count",state.agents.length);
    text("agent-summary",mode==="preview" ? "6 example roles" : state.agents.length ? "Actual worker roster" : "No workers connected");
    $("agent-roster").innerHTML=state.agents.length ? state.agents.map((item,index)=>'<button class="agent-entry" data-agent="'+esc(item.id)+'" aria-pressed="'+(String(item.id)===String(selectedAgent))+'"><span class="agent-avatar">'+creature(index)+'</span><span class="agent-label"><strong>'+esc((item.name || titleCase(item.role)).replace(/^The /,""))+'</strong><small>'+esc(mode==="preview" ? item.role : titleCase(item.role))+'</small></span><span class="agent-dot '+(["running","browsing","preview_browsing"].includes(item.status)?"active":"")+'" aria-hidden="true"></span></button>').join("") : '<p class="empty-notebook">Connect a research brain and browser, then start a mission to create the agent roster.</p>';
    const selected=$("agent-filter").value;
    $("agent-filter").innerHTML='<option value="all">All agents</option>'+state.agents.map(item=>'<option value="'+esc(item.id)+'">'+esc(agentName(item.id))+'</option>').join("");
    if(state.agents.some(item=>String(item.id)===selected)) $("agent-filter").value=selected;
  }
  function illustrativeArticle(source) {
    if(!source) return '<div class="preview-page"><p class="preview-meta">No example source selected.</p></div>';
    const arxiv=sourceUrl(source).includes("arxiv.org"), header=arxiv?"arXiv":"Research archive";
    return '<div class="preview-page"><div class="preview-document-header"><b>'+header+'</b><span>Illustrative article</span></div><article class="preview-article"><div class="preview-meta">'+esc(source.version || "Example source")+'<br>Public reference · '+esc(source.license || "Rights unknown")+'</div><h1 class="preview-title" data-preview-section="0">'+esc(source.title)+'</h1><p class="preview-authors">'+esc(source.authors || "Authors recorded at source")+'</p><h2>Research summary</h2><p data-preview-section="1">'+esc(source.summary || "No source summary recorded.")+'</p><div class="preview-callout"><strong>Question for the record</strong><p data-preview-section="2">'+esc(source.limitation || "Which observations would distinguish the claim from a competing explanation?")+'</p></div><h2>Provenance before conclusions</h2><p data-preview-section="3">The research record keeps the document version, source URL and licensing decision together. Agent notes are separate from original source text.</p><p class="preview-document-footer">Simulated browser. This is an original illustrative layout. Summaries are paraphrases; source full text is not reproduced.</p></article></div>';
  }
  function cancelPreviewScroll(reset=false) {
    if(previewScrollFrame!==null)cancelAnimationFrame(previewScrollFrame);
    previewScrollFrame=null;
    if(reset){previewScrollKey="";previewScrollY=0;}
  }
  function previewPassageLines(target,display) {
    if(!target || typeof document.createRange!=="function")return [];
    const range=document.createRange();range.selectNodeContents(target);
    const origin=display.getBoundingClientRect(),lines=[];
    for(const rect of Array.from(range.getClientRects()).slice(0,128)) {
      const x=Math.max(0,rect.left-origin.left),y=Math.max(0,rect.top-origin.top);
      const right=Math.min(display.clientWidth,rect.right-origin.left),bottom=Math.min(display.clientHeight,rect.bottom-origin.top);
      if(right>x && bottom>y && bottom-y>=5)lines.push({x,y,width:right-x,height:bottom-y});
      if(lines.length===12)break;
    }
    return lines;
  }
  function paintPreviewViewport(page,target,scroll,phase,settled=true) {
    const display=$("browser-display"),selected=agent(),height=Math.max(display.clientHeight,page.scrollHeight);
    page.style.transform="translateY(-"+scroll+"px)";
    const lines=settled?previewPassageLines(target,display):[];
    let focus=null;
    if(lines.length && selected?.preview_inspection_id) {
      const largest=lines.reduce((best,line)=>line.width*line.height>best.width*best.height?line:best);
      const note_id=phase===2?selected.preview_focus_note_id:null;
      focus={...largest,lines,kind:note_id?"preview_passage":"preview_inspection",inspection_id:selected.preview_inspection_id};
      if(note_id)focus.note_id=note_id;
    }
    frameTelemetry={viewport:{viewport_width:display.clientWidth,viewport_height:display.clientHeight,scroll_x:0,scroll_y:scroll,document_width:display.clientWidth,document_height:height},focus,
      frameKey:"preview-"+String(selected?.id)+"-"+String(selected?.step)+"-"+scroll,documentKey:"preview-"+String(selected?.id)+"-"+String(selected?.source_id),scrolling:!settled};
    renderFrameTelemetry();
  }
  function renderPreviewViewport() {
    if(mode!=="preview" || currentView!=="research") return;
    const display=$("browser-display"), page=$("preview-frame").querySelector(".preview-page");
    if(!page || !display.clientWidth || !display.clientHeight) {resetFrameTelemetry();return;}
    const selected=agent(),phase=Math.max(0,Math.min(3,Number(selected?.preview_scroll_phase) || 0));
    const target=page.querySelector('[data-preview-section="'+[1,2,2,3][phase]+'"]');
    const height=Math.max(display.clientHeight,page.scrollHeight),scroll=Math.round(Math.max(0,Math.min(height-display.clientHeight,(target?.offsetTop || 0)-95)));
    const key=[selected?.id,selected?.source_id,phase,height,display.clientWidth,display.clientHeight].join(":");
    if(document.hidden){cancelPreviewScroll();return;}
    if(state.mission.status!=="running" && previewScrollKey) {
      cancelPreviewScroll();paintPreviewViewport(page,target,Math.max(0,Math.min(height-display.clientHeight,previewScrollY)),phase);return;
    }
    if(previewScrollFrame!==null && previewScrollKey===key)return;
    const moving=state.mission.status==="running" && !document.hidden && !motionReduced();
    if(moving && Math.abs(scroll-previewScrollY)>2) {
      cancelPreviewScroll();previewScrollKey=key;
      const from=Math.max(0,Math.min(height-display.clientHeight,previewScrollY)),started=performance.now();
      const tick=now=>{
        if(mode!=="preview" || currentView!=="research" || state.mission.status!=="running" || document.hidden){cancelPreviewScroll();return;}
        const p=Math.min(1,Math.max(0,(now-started)/650)),eased=p*p*(3-2*p);
        previewScrollY=Math.round(from+(scroll-from)*eased);
        paintPreviewViewport(page,target,previewScrollY,phase,p===1);
        if(p<1)previewScrollFrame=requestAnimationFrame(tick);else previewScrollFrame=null;
      };
      previewScrollFrame=requestAnimationFrame(tick);
    } else {
      cancelPreviewScroll();previewScrollKey=key;previewScrollY=scroll;
      paintPreviewViewport(page,target,scroll,phase);
    }
  }
  function renderBrowser() {
    const selected=agent(), source=currentSource(), index=Math.max(0,state.agents.findIndex(item=>String(item.id)===String(selectedAgent)));
    text("browser-agent-name",selected ? agentName(selected.id) : "No agent selected");
    text("browser-agent-role",selected ? (mode==="preview" ? selected.role : titleCase(selected.role)) : "");
    text("browser-url",selected?.current_url || sourceUrl(source) || "Waiting for a browser session");
    text("browser-session-status",mode==="preview" ? "Simulated session" : selected ? titleCase(selected.status || "Connecting") : "No session");
    text("last-observation",selected?.last_observation || [...state.notes].reverse().find(note=>note.agent_id===selected?.id)?.text || selected?.goal || "Observations appear when an agent records them.");
    const marker=$("selected-creature");
    if(marker.dataset.agent!==String(selected?.id || "")) {
      window.ObservatoryMotion?.stop(true);marker.dataset.agent=String(selected?.id || "");
      marker.innerHTML=selected ? creature(index) : "";
    }
    $("selected-creature").hidden=!selected || mode==="connected" && !frameObjectUrl;
    $("preview-frame").hidden=mode!=="preview";$("frame-stamp").hidden=mode==="connected" && !frameObjectUrl;
    text("frame-stamp",mode==="preview" ? "Simulated article layout" : "Actual browser capture");
    if(mode==="preview") {
      $("live-frame").hidden=true;$("browser-empty").hidden=true;
      if(lastPreviewSource!==(source?.id || "")) { cancelPreviewScroll(true);$("preview-frame").innerHTML=illustrativeArticle(source);lastPreviewSource=source?.id || ""; }
      renderPreviewViewport();
      text("frame-description","Example agent-controlled scroll · no browser session");
    } else {
      $("preview-frame").innerHTML="";lastPreviewSource="";
      $("browser-empty").hidden=!!frameObjectUrl;
      if(frameSelection!==String(selectedAgent)) clearLiveFrame();
      if(!selected) text("frame-description","No actual browser session. Start a connected research mission.");
      renderFrameTelemetry();refreshFrame();
    }
    $("browser-inspect").disabled=!source;$("capture-inspect").disabled=!source;$("bookmark-source").disabled=!source;
    text("bookmark-source",source && (source.bookmarked || sourceBookmarks.has(String(source.id))) ? "Bookmarked ★" : "Bookmark source ☆");
  }
  function renderNotebook() {
    text("notebook-scope",selectedAgent ? agentName(selectedAgent)+"’s record" : "Awaiting a researcher");
    const source=currentSource();text("note-source-label",source ? short(source.title,39) : "No source selected");
    $("note-form").querySelector("button").disabled=!source || !canMutate();
    $("note-text").disabled=!source || !canMutate();
    $("note-text").placeholder=mode==="connected" && !ownerToken ? "Owner access is required to add a note." : "A question, caveat, or observation…";
    const capturedId=frameTelemetry?.focus?.note_id;
    const capture=state.notes.find(note=>note.id===capturedId && String(note.agent_id)===String(selectedAgent) && (mode==="preview" || note.support_verified===true)) || [...state.notes].reverse().find(note=>String(note.agent_id)===String(selectedAgent) && note.source_id===source?.id && (mode==="preview" || note.support_verified===true));
    const receipt=$("notebook-capture");receipt.hidden=!capture;receipt.dataset.noteId=capture?String(capture.id):"";
    text("capture-label",mode==="preview"?"Example saved note":"Latest evidence note");
    text("capture-text",capture?capture.passage || capture.supporting_passage || capture.text:"");
    let entries=[];
    if(notebook==="decisions") {
      entries=[...state.events].reverse().filter(event=>event.agent_id===selectedAgent && /decision|action|memory/.test(event.type || "")).slice(0,7).map(event=>({text:event.message || event.text,created_at:event.created_at,label:mode==="preview"?"Example decision":"Action explanation",source_id:event.data?.source_id}));
      if(!entries.length && agent()?.goal) entries=[{text:agent().goal,label:mode==="preview"?"Example next action":"Next action",source_id:source?.id}];
    } else if(notebook==="notes") {
      entries=[...state.notes].reverse().filter(note=>note.agent_id===selectedAgent).slice(0,7).map(note=>({...note,label:titleCase(note.type || "Note")+(mode==="preview"?" · example":"")}));
    } else {
      entries=[...state.notes].reverse().filter(note=>note.agent_id===selectedAgent && (note.passage || note.supporting_passage || note.evidence_passage)).map(note=>({...note,text:note.passage || note.supporting_passage || note.evidence_passage,label:note.support_verified?"Passage provenance checked":"Unverified passage"}));
      if(mode==="preview")entries=entries.map(entry=>({...entry,label:"Illustrated passage · example paraphrase"}));
      if(mode==="preview" && source && !entries.length)entries=[{text:source.summary,label:"Paraphrased summary · example",source_id:source.id}];
    }
    $("notebook-content").innerHTML=entries.length ? entries.map(entry=>'<article class="notebook-entry'+(entry.id===highlightedNoteId?' is-visible-passage':'')+'"'+(entry.id?' data-note-id="'+esc(entry.id)+'"':'')+'><div class="entry-meta"><span>'+esc(entry.label)+(entry.id===highlightedNoteId?' · shown in browser':'')+'</span><time>'+esc(clock(entry.created_at))+'</time></div><p>'+esc(entry.text || "No authored text.")+'</p>'+(sourceById(entry.source_id)?'<button class="source-ref" data-source="'+esc(entry.source_id)+'">'+esc(short(sourceById(entry.source_id).title,48))+'</button>':"")+'</article>').join("") : '<p class="empty-notebook">'+(notebook==="extracts"?"No public source passages are attached to this agent. Full documents remain in the private corpus; a summary is not a quotation.":"This agent has not recorded "+esc(notebook)+" yet.")+'</p>';
    all("[data-notebook]").forEach(button=>{button.classList.toggle("active",button.dataset.notebook===notebook);button.setAttribute("aria-selected",String(button.dataset.notebook===notebook));});
  }
  function renderEvents() {
    const events=[...state.events].reverse().filter(event=>!eventOnlyAgent || event.agent_id===selectedAgent);
    text("events-caption",mode==="preview" ? "Example events · no actual crawling" : connected ? "Actual worker events · "+events.length+" shown" : "Backend unavailable · no simulated events");
    text("event-filter-toggle",eventOnlyAgent ? "Show all agents" : "Selected agent only");
    $("event-stream").innerHTML=events.length ? events.slice(0,50).map(event=>'<article class="event-row"><time datetime="'+esc(event.created_at)+'">'+esc(clock(event.created_at))+'</time><span class="event-agent">'+esc(agentName(event.agent_id))+'</span><span class="event-type">'+esc(String(event.type || "event").replace(/^agent\./,""))+'</span><p>'+esc(event.message || event.text || "Event recorded")+'</p></article>').join("") : '<p class="event-empty">No events match this view. '+(mode==="connected"?"The research worker will append its actual record here.":"Advance the preview to add an example event.")+'</p>';
  }
  function filteredSources() {
    const query=$("evidence-search").value.trim().toLowerCase(), rights=$("rights-filter").value, filter=$("agent-filter").value;
    return state.sources.filter(source=>{
      const haystack=[source.title,sourceUrl(source),source.summary,...list(source.topics)].join(" ").toLowerCase();
      return (!query || haystack.includes(query)) && (filter==="all" || String(source.agent_id)===filter) && (rights==="all" || rights==="eligible" && eligible(source) || rights==="reference" && rightsClass(source)==="reference" || rights==="review" && rightsClass(source)==="review" || rights==="bookmarked" && (source.bookmarked || sourceBookmarks.has(String(source.id))));
    });
  }
  function curationReviews() {
    return list(state.curation_reviews).filter(review=>review && typeof review==="object").slice().sort((a,b)=>(Date.parse(b.created_at) || 0)-(Date.parse(a.created_at) || 0));
  }
  function curationDecision(decision) {
    return ({accepted:["eligible","Accepted by automated review"],manual_review:["review","Awaiting operator review"],rejected:["reference","Rejected by automated review"]})[decision] || ["review","Automated decision not recorded"];
  }
  function sourceCurationReview(source) {
    const receipt=curationReviews().find(review=>String(review.source_id)===String(source.id));
    if(receipt)return receipt;
    const quality=source.quality_review;
    return quality?.reviewer_kind==="automated" ? {...quality,decision:quality.decision || ({approved:"accepted",pending:"manual_review",rejected:"rejected"})[quality.status],model:quality.model || quality.reviewed_by} : null;
  }
  function automatedReviewBadge(source) {
    const review=sourceCurationReview(source);if(!review)return "";
    let [style,label]=curationDecision(review.decision);
    if(review.decision==="accepted") {
      const quality=source.quality_review, prior=quality?.reviewer_kind!=="automated" || ["rejected","quarantined"].includes(source.review_status) || quality?.status && quality.status!=="approved";
      if(prior){style="review";label="Prior automated acceptance";}
      else if(!eligible(source)){style="review";label="Automated review · eligibility pending";}
    }
    return '<span class="tag '+style+' automated-review-tag">'+esc((mode==="preview"?"Simulated · ":"")+label)+'</span>';
  }
  function curationRecoveryAvailable(reviewId) {
    const runtime=state.curation_runtime || {};
    return mode==="connected" && connected && canMutate() && !busy && ["paused","faulted","stopped"].includes(state.mission.status) && ["in_flight","awaiting_operator"].includes(runtime.call_state) && typeof reviewId==="string" && !!reviewId && runtime.review_id===reviewId;
  }
  function curationRecoveryMarkup(runtime) {
    if(mode==="preview" || !["paused","faulted","stopped"].includes(state.mission.status) || !["in_flight","awaiting_operator"].includes(runtime.call_state))return "";
    const enabled=curationRecoveryAvailable(runtime.review_id), source=sourceById(runtime.source_id);
    return '<div class="curation-recovery"><div><strong>Interrupted review needs operator attention</strong><p>'+esc(runtime.active_stage?'Stage: '+titleCase(runtime.active_stage)+'. ':"")+'Reconcile any pending payment before authorizing another review attempt. Completed review receipts are preserved.</p>'+(source?'<button class="text-button" data-source="'+esc(source.id)+'">'+esc(source.title || sourceUrl(source))+'</button>':"")+'</div><button type="button" class="quiet-button" data-curation-recover="'+esc(runtime.review_id || "")+'"'+(enabled?"":" disabled")+'>Review recovery</button></div>';
  }
  function renderAutomatedCuration() {
    const reviews=curationReviews(), requested=state.settings?.auto_curation_enabled===true, enabled=requested && state.settings?.auto_curation_policy_ack===autoCurationPolicyAck, needsUpdate=requested && !enabled, running=state.mission.status==="running";
    const runtime=state.curation_runtime || {}, interrupted=["in_flight","awaiting_operator"].includes(runtime.call_state) && ["paused","faulted","stopped"].includes(state.mission.status);
    const status=mode==="preview"?"Simulated review":!connected?"Backend unavailable":interrupted?"Operator attention required":needsUpdate?"Policy update required":!enabled?"Disabled":running?"Enabled during research":"Waiting for research";
    const counts={accepted:0,manual_review:0,rejected:0};
    reviews.forEach(review=>{if(Object.prototype.hasOwnProperty.call(counts,review.decision))counts[review.decision]++;});
    const latest=reviews[0], source=latest && sourceById(latest.source_id), [style,label]=curationDecision(latest?.decision);
    const explanation=mode==="preview"?"These are simulated records. Changing the setup toggle does not run a real review.":!connected?"Connect the backend to see original-document review receipts.":needsUpdate?"The saved review policy needs renewal for the broader consciousness scope. Automated review is idle. Open Operator setup, enable the updated policy and save setup.":enabled?"Two blind review passes classify original documents by subject and evidence type while the mission runs. Scientific findings, philosophical arguments and religious interpretations stay distinct. Source rights, extraction and corpus exclusions still apply. Q&A approval remains separate.":"Enable automated original-document curation in Operator setup. Collected sources remain available for manual review.";
    $("automated-curation-summary").innerHTML='<div class="section-top"><h3 id="automated-curation-title">Automated original-document review</h3><span class="tag '+(enabled && mode!=="preview" && connected && !interrupted?"eligible":"review")+'">'+esc(status)+'</span></div><p class="curation-explanation">'+esc(explanation)+'</p><div class="curation-counts"><span><strong>'+counts.accepted+'</strong> accepted receipts</span><span><strong>'+counts.manual_review+'</strong> await review</span><span><strong>'+counts.rejected+'</strong> rejected receipts</span></div>'+(latest?'<div class="curation-latest"><span class="tag '+style+'">'+esc((mode==="preview"?"Simulated · ":"")+label)+'</span><span>'+esc(latest.model || "Reviewer model not recorded")+' · '+esc(clock(latest.created_at))+'</span>'+(source?'<button class="text-button" data-source="'+esc(source.id)+'">'+esc(source.title || sourceUrl(source))+'</button>':"")+'<p>'+esc(latest.rationale || list(latest.reasons).map(titleCase).join("; ") || "No review rationale recorded.")+'</p></div>':'<p class="curation-empty">'+esc(mode==="preview"?"No simulated automated review receipts. No model call was made.":"No automated review receipts have been recorded.")+'</p>');
    $("automated-curation-summary").insertAdjacentHTML("beforeend",curationRecoveryMarkup(runtime));
  }
  function renderEvidence() {
    const sources=filteredSources();text("evidence-count",state.sources.length);text("source-results",sources.length+" of "+state.sources.length+" records");
    $("evidence-summary").innerHTML='<span><strong>'+state.sources.filter(eligible).length+'</strong> '+(mode==="preview"?"example eligible sources":"eligible sources")+'</span><span><strong>'+state.sources.filter(source=>rightsClass(source)==="review").length+'</strong> need review</span><span><strong>'+state.sources.filter(source=>rightsClass(source)==="reference").length+'</strong> reference only</span><span>Original documents ≠ agent notes</span>';
    renderAutomatedCuration();
    $("source-ledger").innerHTML=sources.map(source=>'<tr><td><button class="source-title" data-source="'+esc(source.id)+'">'+esc(source.title || sourceUrl(source))+'</button><div class="source-meta">'+esc(sourceUrl(source))+'<br>'+esc(source.version || source.provenance?.method || "Version not recorded")+(mode==="preview"?" · example capture":"")+'</div></td><td>'+esc(agentName(source.agent_id))+'</td><td><span class="tag '+rightsClass(source)+'">'+esc(rightsLabel(source))+'</span><span class="source-license">'+esc(source.license || "License unknown")+'</span></td><td><span class="tag">'+esc(titleCase(source.review_status || "Pending"))+'</span>'+automatedReviewBadge(source)+'</td><td><button class="icon-button" data-source="'+esc(source.id)+'" aria-label="Inspect '+esc(source.title || "source")+'">↗</button></td></tr>').join("");
    $("source-empty").hidden=!!sources.length;
  }
  function datasetName(dataset) { return dataset.name || "Corpus "+short(dataset.id,25); }
  function manifestRows(dataset) {
    const records=list(dataset.manifest?.source_records);
    return list(dataset.source_ids || dataset.manifest?.source_ids).map(id=>{const record=records.find(item=>String(item.id)===String(id));return {...sourceById(id),...record,id,title:record?.title || sourceById(id)?.title || id};});
  }
  function corpusAuditMarkup(dataset) {
    if(!dataset)return '<p class="method-note">The next snapshot will report perspective coverage, duplicate removal and quality exclusions.</p>';
    const audit=dataset.coverage_audit || dataset.manifest?.coverage_audit, gate=dataset.quality_gate || dataset.manifest?.quality_gate;
    if(mode==="preview" || !audit)return '<h3>Review before training</h3><p class="method-note">'+esc(mode==="preview"?"This simulated metadata snapshot contains no original training text or measured coverage. Real snapshots distinguish consciousness science, philosophy, metaphysics, contemplation and AI. The machine-consciousness coverage floor requires reviewed supporting, skeptical and uncertain perspectives; scientific material and extraction checks are also required.":"This older snapshot has no current quality audit. Create a new reviewed snapshot before training.")+'</p>';
    const train=audit.splits?.train || audit.train || {}, stances=train.stances || {}, shares=train.stance_character_shares || {};
    const rows=['supportive','skeptical','uncertain','mixed','methodological'].map(stance=>'<div class="coverage-row"><span>'+esc(titleCase(stance))+'</span><div class="coverage-meter"><i style="width:'+Math.max(0,Math.min(100,Number(shares[stance] || 0)*100))+'%"></i></div><span>'+esc(stances[stance] || 0)+' copies</span></div>').join('');
    return '<div class="section-top"><h3>Corpus quality</h3><span class="tag '+(gate?.ready?'eligible':'review')+'">'+esc(gate?.ready?'Coverage floor met':'Coverage needs review')+'</span></div><h4 class="coverage-heading">Machine-consciousness perspectives</h4>'+rows+coverageBucketsMarkup(train.topic_domains,'Reviewed research areas',topicDomains)+coverageBucketsMarkup(train.evidence_kinds,'Basis of claims',evidenceKinds)+'<p class="method-note">Bars show machine-perspective original-text character share in the training split. General philosophy or religious accounts do not supply machine perspectives automatically. Coverage is a presence check, not equal weighting or scientific validation. '+esc(dataset.counts?.deduplicated_sources || 0)+' duplicate copies removed; their source lineage remains recorded.</p>'+(list(gate?.reasons).length?'<p class="rights-explanation">'+esc(list(gate.reasons).map(titleCase).join('; '))+'</p>':'');
  }
  function coverageBucketsMarkup(buckets,heading,labels) {
    if(!buckets || typeof buckets!=="object" || Array.isArray(buckets))return "";
    const entries=Object.entries(buckets).filter(([,count])=>Number.isSafeInteger(count) && count>=0);
    if(!entries.length)return "";
    return '<div class="scope-coverage"><h4 class="coverage-heading">'+esc(heading)+'</h4><dl>'+entries.map(([key,count])=>'<div><dt>'+esc(labels.find(([id])=>id===key)?.[1] || titleCase(key))+'</dt><dd>'+esc(count)+' documents</dd></div>').join('')+'</dl></div>';
  }
  function renderDatasets() {
    text("dataset-count",state.datasets.length);text("snapshot-create",mode==="preview" ? "Create preview snapshot" : "Create curated snapshot");
    $("snapshot-create").disabled=busy || mode==="connected" && !connected;
    $("dataset-list").innerHTML=state.datasets.length ? [...state.datasets].reverse().map(dataset=>'<button class="dataset-entry '+(String(dataset.id)===String(selectedDataset)?"selected":"")+'" data-dataset="'+esc(dataset.id)+'"><span class="tag">'+esc(mode==="preview"?"Example candidate":titleCase(dataset.status || "Candidate"))+'</span><strong>'+esc(datasetName(dataset))+'</strong><small>'+esc(dateLabel(dataset.created_at))+'<br>'+esc(dataset.counts?.original_documents ?? list(dataset.source_ids).length)+' original documents · '+esc(dataset.counts?.synthetic_examples ?? 0)+' instruction examples</small></button>').join("") : '<p class="empty-notebook">No curated snapshots.<br>Review source rights, then create the first immutable candidate.</p>';
    const dataset=state.datasets.find(item=>String(item.id)===String(selectedDataset));
    $("corpus-audit").innerHTML=corpusAuditMarkup(dataset);
    $("manifest-download").disabled=!dataset;
    $("dataset-export").disabled=!dataset || mode==="preview" || !canMutate() || busy;
    text("manifest-state",dataset ? mode==="preview" ? "Example snapshot" : (dataset.immutable ? "Immutable candidate" : titleCase(dataset.status)) : "Awaiting snapshot");
    text("manifest-name",dataset ? datasetName(dataset) : "No curated snapshot selected");
    all("[data-manifest]").forEach(button=>button.classList.toggle("active",button.dataset.manifest===manifestView));
    if(!dataset) {
      $("manifest-meta").innerHTML="";
      $("manifest-content").innerHTML='<div class="empty-panel"><h3>Awaiting a curated snapshot</h3><p>Source licensing, provenance and quality decisions must accompany the corpus.</p></div>';return;
    }
    $("manifest-meta").innerHTML='<div><strong>Revision fingerprint'+(mode==="preview"?" · example":"")+'</strong><code>'+esc(short(dataset.manifest_hash || dataset.corpus_hash,32))+'</code></div><div><strong>Source families</strong>'+esc(dataset.counts?.original_documents ?? list(dataset.source_ids).length)+' documents</div><div><strong>Held-out material</strong>'+esc(dataset.counts?.validation_documents ?? list(dataset.heldout_family_ids).length)+' records</div><div><strong>Instruction examples</strong>'+esc(dataset.counts?.synthetic_examples ?? 0)+' approved records</div>';
    const rows=manifestRows(dataset);
    if(manifestView==="manifest") {
      $("manifest-content").innerHTML=rows.length ? rows.map((record,index)=>'<div class="manifest-row"><span>'+String(index+1).padStart(2,"0")+'</span><div><strong>'+esc(record.title)+'</strong><small>'+esc(record.canonical_url || sourceUrl(record) || record.id)+'<br>'+esc(record.version || "Version in manifest")+' · '+esc(record.license || "License in source record")+'</small></div><span class="tag eligible">'+(mode==="preview"?"Example cleared":"Included")+'</span></div>').join("") : '<p class="empty-notebook">This candidate contains no included source records. Review the exclusions before training.</p>';
    } else if(manifestView==="excluded") {
      const excluded=list(dataset.manifest?.excluded_sources || dataset.excluded_sources);
      $("manifest-content").innerHTML=excluded.length ? excluded.map(record=>'<div class="manifest-row"><span>×</span><div><strong>'+esc(sourceById(record.source_id)?.title || record.source_id)+'</strong><small>'+esc(list(record.reasons).map(titleCase).join("; ") || "Excluded by curation policy")+'</small></div><span class="tag reference">Excluded</span></div>').join("") : '<p class="empty-notebook">No source exclusions recorded for this snapshot.</p>';
    } else {
      const index=state.datasets.findIndex(item=>String(item.id)===String(selectedDataset)), previous=state.datasets[index-1], oldRows=previous?manifestRows(previous):[], oldIds=new Set(oldRows.map(record=>String(record.id))), currentIds=new Set(rows.map(record=>String(record.id)));
      const added=rows.filter(record=>!oldIds.has(String(record.id))), removed=oldRows.filter(record=>!currentIds.has(String(record.id)));
      const changed=rows.filter(record=>{const old=oldRows.find(item=>String(item.id)===String(record.id));return old && JSON.stringify(old.rights_evidence)!==JSON.stringify(record.rights_evidence);});
      $("manifest-content").innerHTML='<p class="rights-explanation">'+esc(previous?"Compared with "+datasetName(previous)+".":"First snapshot: all included source records are additions.")+'</p>'+[...added.map(record=>({record,sign:"+",label:"Added"})),...removed.map(record=>({record,sign:"−",label:"Removed"})),...changed.map(record=>({record,sign:"~",label:"Rights record changed"}))].map(({record,sign,label})=>'<div class="manifest-row"><span>'+sign+'</span><div><strong>'+esc(record.title)+'</strong><small>'+esc(label)+'</small></div><span class="tag">'+esc(label)+'</span></div>').join("")+(!added.length&&!removed.length&&!changed.length?'<p class="empty-notebook">No source membership or recorded rights changes.</p>':"");
    }
  }
  function renderTraining() {
    const datasetValue=$("training-dataset").value, jobValue=selectedJob, stage=$("training-stage").value, settings=mergeSettings(defaults,state.settings), parents=state.jobs.filter(job=>job.stage==="cpt" && (job.status==="passed" || mode==="preview" && job.status==="preview_validated"));
    $("training-dataset").innerHTML='<option value="">'+(state.datasets.length?"Choose a curated snapshot":"Awaiting curated snapshot")+'</option>'+state.datasets.map(dataset=>'<option value="'+esc(dataset.id)+'">'+esc(datasetName(dataset))+'</option>').join("");
    if(state.datasets.some(item=>String(item.id)===datasetValue)) $("training-dataset").value=datasetValue;else if(selectedDataset) $("training-dataset").value=selectedDataset;
    const parentValue=$("training-parent").value;
    $("training-parent").innerHTML='<option value="">Choose a completed job</option>'+parents.map(job=>'<option value="'+esc(job.id)+'">'+esc(job.name || short(job.id,31))+'</option>').join("");
    if(parents.some(job=>String(job.id)===parentValue)) $("training-parent").value=parentValue;
    $("training-parent-label").hidden=stage!=="sft";
    $("training-job-select").innerHTML='<option value="">'+(state.jobs.length?"Choose a job":"No jobs")+'</option>'+state.jobs.map(job=>'<option value="'+esc(job.id)+'">'+esc(job.name || short(job.id,30))+'</option>').join("");
    if(jobValue) $("training-job-select").value=jobValue;
    text("recipe-baseline",settings.hf_base_model || "Not configured");
    const revision=String(settings.hf_base_revision || "");
    text("recipe-revision",revision ? /^[a-f0-9]{40}$/i.test(revision) ? revision.slice(0,12) : short(revision,24) : "Resolve at preparation");
    $("recipe-revision").title=revision || "The backend resolves and records an immutable commit before submitting a real job.";
    text("recipe-adaptation",adaptationLabel(settings.training_mode));
    text("recipe-path","Base checkpoint → CPT with "+adaptationLabel(settings.training_mode)+" → separate approved SFT.");
    $("recipe-attribution").hidden=settings.hf_base_model!==defaults.hf_base_model;
    text("recipe-publication",titleCase(settings.publish_policy || "Private")+" · owner controlled");
    text("training-connection",mode==="preview" ? "Preview · no GPU job" : settings.training_enabled ? "Training enabled by owner" : "Training disabled");
    const approved=!!settings.synthetic_training_approved && !!settings.provider_policy_reference;
    const chosen=state.datasets.find(item=>String(item.id)===$("training-dataset").value), qualityReady=mode==="preview" || (chosen?.quality_gate || chosen?.manifest?.quality_gate)?.ready===true;
    $("training-start").disabled=busy || !canMutate() || !($("training-dataset").value) || !qualityReady || mode==="connected" && !settings.training_enabled || stage==="sft" && (!approved || !$("training-parent").value);
    text("training-start",mode==="preview" ? "Validate preview job" : stage==="sft" ? "Submit approved SFT job" : "Submit pretraining job");
    const running=state.jobs.find(job=>["running","submitting","submitted","preparing"].includes(job.status));
    $("worker-state").innerHTML='<span aria-hidden="true">◇</span><div><strong>'+esc(mode==="preview"?"No training is running":running?"Actual job: "+titleCase(running.status):settings.training_enabled?"Ready for owner submission":"Training is disabled")+'</strong><p>'+esc(mode==="preview" ? "Preview validation demonstrates the flow. It does not allocate a GPU or create weights." : stage==="sft"&&!approved?"Enable approved instruction examples and record the teacher provider policy in setup before SFT." : running?"The worker reports its status and measured results in the job record." : "GPU access, Hugging Face permissions and a sealed dataset are checked by the backend before submission.")+'</p></div>';
    if(mode==="connected" && !running && chosen && !qualityReady) {
      const reasons=list((chosen.quality_gate || chosen.manifest?.quality_gate)?.reasons);
      $("worker-state").innerHTML='<span aria-hidden="true">◇</span><div><strong>Corpus review incomplete</strong><p>'+esc(reasons.length?reasons.map(titleCase).join('; '):'Create a new reviewed snapshot with the current corpus policy. Inspect perspective coverage and exclusions in Datasets.')+'</p></div>';
    }
    const job=state.jobs.find(item=>String(item.id)===String(selectedJob));
    if(!job) {
      $("job-status").innerHTML='<span class="tag">Awaiting curated snapshot</span>';
      text("job-logs","No training job has been submitted.\nA real worker must report logs and measured results.");$("job-footer").innerHTML="";return;
    }
    $("job-status").innerHTML='<span class="tag '+(/failed|error/.test(job.status)?"error":"")+'">'+esc(mode==="preview"?"Preview validation":titleCase(job.status))+'</span><p>'+esc(titleCase(job.stage || "cpt"))+' · '+esc(short(job.snapshot_id || job.manifest?.snapshot_id,24))+'</p>';
    const logs=list(job.logs).map(line=>typeof line==="string"?line:JSON.stringify(line)), eventLogs=state.events.filter(event=>event.run_id===job.id && /training|worker/.test(event.type || "")).map(event=>"["+clock(event.created_at)+"] "+(event.message || ""));
    text("job-logs",(logs.length?logs:eventLogs).join("\n") || "No worker logs reported yet.");
    $("job-footer").innerHTML='<span>'+esc(mode==="preview"?"No weights or measured metrics exist.":"Created "+dateLabel(job.created_at))+'</span>'+(['running','submitted','preparing','submitting'].includes(job.status)?'<button class="text-button" data-cancel-job="'+esc(job.id)+'">Cancel job</button>':"")+(mode==="connected" && ['failed','cancelled'].includes(job.status)?'<button class="text-button" data-retry-job="'+esc(job.id)+'"'+(busy || !canMutate()?' disabled':"")+'>'+esc(job.status==="failed"?"Retry failed job":"Retry cancelled job")+'</button>':"");
    if(mode==="connected" && job.metrics && Object.keys(job.metrics).length) $("job-footer").innerHTML+='<details><summary>Measured worker results</summary><pre class="manifest-json">'+esc(JSON.stringify(job.metrics,null,2))+'</pre></details>';
  }
  function renderCheckpoints() {
    $("checkpoint-list").innerHTML=state.checkpoints.length ? state.checkpoints.map(checkpoint=>'<article class="checkpoint-row"><div><h3>'+esc(checkpoint.name || checkpoint.repo_id || short(checkpoint.id,35))+'</h3><p>'+esc(checkpoint.base_model || checkpoint.model_id || "Model lineage in checkpoint record")+'<br>'+esc(short(checkpoint.revision || checkpoint.artifact_revision || "No published weights",50))+'</p></div><div><span>Calibration</span><span class="tag '+(checkpoint.calibration?.passed?"eligible":"review")+'">'+esc(checkpoint.baseline?"Baseline retained":checkpoint.calibration?.passed?"Measured · passed":"Calibration required")+'</span></div><div><span>Selection</span><span class="tag">'+esc(checkpoint.baseline?"Control":needsChatSft(checkpoint)?"SFT required for chat":titleCase(checkpoint.status || "Candidate"))+'</span></div><button class="quiet-button" data-checkpoint="'+esc(checkpoint.id)+'">Inspect record</button></article>').join("") : '<div class="empty-panel"><h3>No research checkpoint yet</h3><p>The existing chamber baseline remains unchanged. A real training job must produce weights before a candidate appears.</p></div>';
  }
  function needsChatSft(checkpoint) {
    const job=state.jobs.find(item=>String(item.id)===String(checkpoint.run_id));
    const base=checkpoint.base_model || checkpoint.model_id || job?.manifest?.base_model;
    return base===defaults.hf_base_model && (checkpoint.stage || job?.stage)!=="sft";
  }
  function renderConnections() {
    const names={research:"Research brain",brain:"Research brain",browser:"Browser infrastructure",huggingface:"Hugging Face",hf:"Hugging Face",training:"GPU worker"};
    let connections=Array.isArray(state.connections)?state.connections:Object.entries(state.connections || {}).map(([id,value])=>({id,...(typeof value==="object"?value:{status:value})}));
    if(!connections.length) connections=Object.keys({research:1,browser:1,huggingface:1,training:1}).map(id=>({id,status:"Not reported"}));
    $("connection-list").innerHTML=connections.map(item=>'<div class="connection-row"><span>'+esc(item.name || names[item.id] || titleCase(item.id))+'</span><span class="'+(["connected","ready","healthy","ok"].includes(item.status)?"healthy":"")+'">'+esc(item.message || titleCase(item.status || "Not reported"))+'</span></div>').join("");
  }
  function showView(view,focus=false) {
    if(!["research","evidence","datasets","training","checkpoints"].includes(view)) view="research";
    currentView=view;
    if(view!=="research")window.ObservatoryMotion?.stop();
    all(".view").forEach(panel=>panel.hidden=panel.id!=="view-"+view);
    all("[data-view]").forEach(button=>{const active=button.dataset.view===view;button.setAttribute("aria-selected",String(active));button.tabIndex=active?0:-1;});
    if(focus) $("tab-"+view).focus();
    history.replaceState(null,"",location.pathname+location.search+"#"+view);
    if(view==="research") {if(mode==="preview"){renderPreviewViewport();renderNotebook();}else refreshFrame();}
  }
  function openDialog(id) { if(!$(id).open) $(id).showModal(); }
  function openSetup() {
    const fields={api:preferences.api_base,objective:state.mission.objective || preferences.objective || $("setting-objective").value,provider:state.settings.research_provider || preferences.research_provider,model:state.settings.research_model || preferences.research_model,browser:state.settings.browser_provider || preferences.browser_provider,hf:state.settings.hf_namespace || preferences.hf_namespace};
    Object.entries(fields).forEach(([name,value])=>{if(value!==undefined) $("setting-"+name).value=value;});
    all('input[name="mode"]').forEach(input=>input.checked=input.value===mode);
    $("owner-token").value=ownerToken;renderConnections();fillExtraSettings();previousProvider=$("setting-provider").value;syncResearchFields(fields.model);openDialog("setup-dialog");
    refreshPaymentMetadata(true);
  }
  function bookmarkSource(id) {
    const source=sourceById(id);if(!source) return;
    const active=sourceBookmarks.has(String(id)) || !!source.bookmarked;
    if(active){sourceBookmarks.delete(String(id));if(mode==="preview") source.bookmarked=false;}
    else {sourceBookmarks.add(String(id));if(mode==="preview") source.bookmarked=true;}
    try {localStorage.setItem(bookmarkKey,JSON.stringify([...sourceBookmarks]));}catch(_){}
    renderBrowser();renderEvidence();toast(active?"Source bookmark removed from this browser.":"Source bookmarked in this browser.");
  }
  function evidenceText(value) { return typeof value==="object" ? JSON.stringify(value,null,2) : String(value || "No rights evidence recorded."); }
  function sourceAutomatedReviewMarkup(source) {
    const review=sourceCurationReview(source);if(!review)return "";
    const [style,label]=curationDecision(review.decision);
    return '<div class="record-block source-automated-review"><div class="section-top"><h3>Automated review receipt</h3><span class="tag '+style+'">'+esc((mode==="preview"?"Simulated receipt · ":"Receipt · ")+label)+'</span></div><p class="curation-review-model">'+esc(review.model || "Reviewer model not recorded")+(review.created_at?' · '+esc(dateLabel(review.created_at)):"")+'</p><p class="rights-explanation">'+esc(review.rationale || "No review rationale recorded.")+'</p>'+(list(review.reasons).length?'<p class="rights-explanation">Review reasons: '+esc(list(review.reasons).map(titleCase).join("; "))+'</p>':"")+'<p class="review-help">'+esc(mode==="preview"?"This is a simulated model review. It grants no actual training rights.":"This receipt records the automated decision at the time of review, not current eligibility or an operator approval. Later operator decisions, source rights, extraction and corpus policy still determine training eligibility. Q&A approval remains separate.")+'</p></div>';
  }
  function qualityReviewMarkup(source,canReview) {
    const q=source.quality_review || {}, disabled=canReview?"":" disabled";
    const select=(name,value,options)=>'<select name="'+name+'"'+disabled+'>'+options.map(([id,label])=>'<option value="'+id+'"'+(id===value?' selected':'')+'>'+label+'</option>').join("")+'</select>';
    const checkbox=(name,label,checked)=>'<label><input type="checkbox" name="'+name+'"'+(checked?' checked':'')+disabled+'> '+label+'</label>';
    return '<fieldset class="quality-review-fields"><legend>Corpus quality</legend><p class="review-help">Review the exact document. Classify its argument, not whether consciousness is established.</p>'+
      '<label>Topic relevance'+select('topic_relevance',q.topic_relevance || 'uncertain',[['uncertain','Needs assessment'],['relevant','Relevant to the mission'],['unrelated','Unrelated']])+'</label>'+
      '<fieldset class="review-topic-domains"><legend>Research areas · choose at least one</legend>'+topicDomains.map(([id,label])=>checkbox('topic_'+id,label,list(q.topic_domains).includes(id))).join('')+'</fieldset>'+
      '<label>Basis of claims'+select('evidence_kind',q.evidence_kind || '',[['','Choose after reviewing the document'],...evidenceKinds])+'</label>'+
      '<label>Machine-consciousness perspective'+select('evidence_stance',q.evidence_stance || 'uncertain',[['not_applicable','Not applicable · no machine-consciousness claim'],['supportive','Arguments supporting possible machine consciousness'],['skeptical','Arguments questioning machine consciousness'],['uncertain','Unresolved evidence about machine consciousness'],['mixed','Several machine-consciousness perspectives'],['methodological','Machine-consciousness methods and measurement']])+'</label>'+
      '<p class="review-help">Use Not applicable for documents that make no machine-consciousness claim. General philosophy, metaphysics or contemplative accounts can inform the mission without fulfilling the machine-perspective coverage floor.</p>'+
      '<label>Document type'+select('quality_source_type',q.source_type || 'article',[['empirical_paper','Empirical paper'],['theoretical_paper','Theoretical paper'],['review_paper','Review paper'],['technical_report','Technical report'],['article','Article'],['reference','Reference'],['social','Social post']])+'</label>'+
      '<label>Operator reviewer<input name="quality_reviewed_by" value="'+esc(q.reviewer_kind==="automated"?'Operator':q.reviewed_by || 'Operator')+'" maxlength="4000"'+disabled+'></label>'+
      '<label>Quality rationale<textarea name="quality_rationale" rows="3"'+disabled+' placeholder="Relevance, evidence limitations, extraction checks and why this copy belongs in the corpus">'+esc(q.rationale || '')+'</textarea></label>'+
      '<div class="covered-perspectives"><span>For mixed or methodological documents, machine perspectives actually covered:</span>'+['supportive','skeptical','uncertain'].map(stance=>checkbox('covered_'+stance,titleCase(stance),q.evidence_stance!=='not_applicable' && list(q.covered_stances).includes(stance))).join('')+'</div>'+
      checkbox('contains_benchmark','Contains evaluation questions or benchmark answers',source.contains_benchmark || source.contamination_status && source.contamination_status!=='clear')+
      checkbox('chamber_stimulus','Contains Chamber experiment prompts or stimuli',source.chamber_stimulus || source.experimental_stimulus)+
      '<p class="review-help">Extraction: '+esc(source.extraction?.method || 'Legacy capture')+' / '+esc(source.extraction?.quality || 'Unverified')+'. '+esc(list(source.extraction?.warnings).join('; '))+'</p>'+
      checkbox('extraction_review_approved','I checked the extracted text against the original copy',source.extraction_review_status==='approved')+
      '<label>Extraction review evidence<textarea name="extraction_review_evidence" rows="2"'+disabled+' placeholder="Record checks of reading order, tables, formulas and missing pages">'+esc(source.extraction_review_evidence || '')+'</textarea></label></fieldset>';
  }
  function qualityClassificationMarkup(source) {
    const q=source.quality_review || {}, domains=list(q.topic_domains).map(id=>topicDomains.find(([key])=>key===id)?.[1] || titleCase(id));
    const kind=evidenceKinds.find(([id])=>id===q.evidence_kind)?.[1];
    return '<div class="record-block reviewed-classification"><h3>Reviewed classification</h3><dl class="record-grid"><div><dt>Research areas</dt><dd>'+esc(domains.length?domains.join(' · '):'Not classified')+'</dd></div><div><dt>Basis of claims</dt><dd>'+esc(kind || 'Not classified')+'</dd></div><div><dt>Machine-consciousness perspective</dt><dd>'+esc(q.evidence_stance==='not_applicable'?'Not applicable':q.evidence_stance?titleCase(q.evidence_stance):'Not classified')+'</dd></div><div><dt>Review status</dt><dd>'+esc(q.status?titleCase(q.status):'Pending assessment')+(mode==='preview'?' · simulated':'')+'</dd></div></dl><p class="review-help">A research area identifies what the document discusses. The basis of claims identifies the kind of argument; neither establishes its truth.</p></div>';
  }
  function syncSourceReviewFields(form,domainsChanged=false) {
    const stance=form.querySelector('[name="evidence_stance"]'), machine=form.querySelector('[name="topic_machine_consciousness"]');
    if(!stance)return;
    if(domainsChanged && !machine?.checked)stance.value='not_applicable';
    const active=['mixed','methodological'].includes(stance.value), readOnly=stance.disabled;
    form.querySelectorAll('[name^="covered_"]').forEach(input=>{if(!active)input.checked=false;input.disabled=readOnly || !active;});
  }
  function inspectSource(id) {
    const source=sourceById(id);if(!source) return;
    const url=safeUrl(sourceUrl(source)), incompatible=rightsClass(source)==="reference", provenance=source.provenance || {}, canReview=canMutate();
    text("source-dialog-tag",mode==="preview"?"Example source record":"Collected source record");
    $("source-dialog-content").innerHTML='<h2>'+esc(source.title || sourceUrl(source))+'</h2>'+(url?'<a class="source-external" href="'+esc(url)+'" target="_blank" rel="noopener noreferrer">'+esc(url)+' ↗</a>':"")+'<div class="record-grid"><div><dt>Collected by</dt><dd>'+esc(agentName(source.agent_id))+'</dd></div><div><dt>Version / family</dt><dd>'+esc(source.version || source.family_id || "Not recorded")+'</dd></div><div><dt>Rights</dt><dd><span class="tag '+rightsClass(source)+'">'+esc(rightsLabel(source))+'</span><br>'+esc(source.license || "Unknown")+'</dd></div><div><dt>Capture method</dt><dd>'+esc(mode==="preview"?"Illustrative fixture":provenance.method || "Not recorded")+'<br>'+esc(dateLabel(provenance.collected_at || source.created_at))+'</dd></div><div><dt>Content fingerprint</dt><dd>'+esc(short(source.content_hash || provenance.content_sha256,34))+'</dd></div><div><dt>Original document</dt><dd>'+esc(mode==="preview"?"Not reproduced in the preview":source.word_count?source.word_count+" words · full text stored privately":"Full text is private; only metadata is public")+'</dd></div></div><p class="record-summary">'+esc(source.summary || "No public summary recorded.")+'</p>'+(source.limitation?'<div class="record-excerpt">'+esc(source.limitation)+'</div>':"")+sourceAutomatedReviewMarkup(source)+'<div class="record-block"><h3>Rights &amp; corpus review</h3><p class="rights-explanation">'+esc(evidenceText(source.rights_evidence || source.permission_evidence))+'</p>'+(list(source.curation?.reasons).length?'<p class="rights-explanation">Excluded because: '+esc(list(source.curation.reasons).map(titleCase).join("; "))+'</p>':"")+'<form class="review-form" data-review-source="'+esc(id)+'"><label>Review decision<select name="review_status" '+(!canReview?"disabled":"")+'><option value="approved">Approve relevance</option><option value="pending">Keep pending</option><option value="quarantined">Reference only / exclude</option><option value="rejected">Reject</option></select></label><label>License for this exact copy<input name="license" value="'+esc(source.license || "unknown")+'" '+(!canReview?"disabled":"")+'></label><label><input name="license_verified" type="checkbox" '+(source.license_verified?"checked ":"")+(!canReview?"disabled":"")+'> Verified training-compatible rights for this exact copy</label><label>Rights evidence<textarea name="rights_evidence" rows="3" '+(!canReview?"disabled":"")+' placeholder="License URL, version and permission evidence">'+esc(source.rights_evidence?evidenceText(source.rights_evidence):"")+'</textarea></label><label>Review note<textarea name="review_note" rows="2" '+(!canReview?"disabled":"")+' placeholder="Attribution, exclusions, relevance or caveats">'+esc(source.review_note || "")+'</textarea></label>'+qualityReviewMarkup(source,canReview)+'<p class="review-help">'+esc(mode==="preview"?"This records a simulated review. It grants no real rights.":canReview?"The backend validates eligibility; checking a box alone does not clear a source.":"Public inspection is read-only. Open Operator setup to authenticate.")+'</p><button class="primary-button" type="submit" '+(!canReview?"disabled":"")+'>'+esc(mode==="preview"?"Save preview review":"Save source review")+'</button></form></div><div class="dialog-actions"><button class="quiet-button" data-bookmark-source="'+esc(id)+'">'+(sourceBookmarks.has(String(id)) || source.bookmarked?"Remove bookmark":"Bookmark source")+'</button></div>';
    $("source-dialog-content").querySelector('[name="review_status"]').value=["approved","pending","quarantined","rejected"].includes(source.review_status)?source.review_status:incompatible?"quarantined":"pending";
    $("source-dialog-content").querySelector('.record-summary').insertAdjacentHTML('afterend',qualityClassificationMarkup(source));
    syncSourceReviewFields($("source-dialog-content").querySelector('[data-review-source]'));
    openDialog("source-dialog");
  }
  function renderAllNotes() {
    const query=$("notes-search").value.toLowerCase().trim(), bookmarks=$("notes-bookmarked").checked;
    const notes=[...state.notes].reverse().filter(note=>(!query || [note.text,note.question,note.answer,sourceById(note.source_id)?.title].join(" ").toLowerCase().includes(query)) && (!bookmarks || note.bookmarked));
    $("all-notes").innerHTML=notes.length ? notes.map(note=>'<article class="all-note"><div class="entry-meta"><span>'+esc(agentName(note.agent_id))+' / '+esc(titleCase(note.type || "Note"))+(mode==="preview"?" · example":"")+'</span><time>'+esc(clock(note.created_at))+'</time></div><p>'+esc(note.text || note.question || "No note text")+'</p>'+(note.question?'<p class="rights-explanation">Instruction draft: '+esc(note.question)+'<br>'+esc(note.answer || "")+'</p>':"")+'<div class="note-tools">'+(sourceById(note.source_id)?'<button class="source-ref text-button" data-source="'+esc(note.source_id)+'">'+esc(short(sourceById(note.source_id).title,47))+'</button>':'<span class="review-help">No linked source</span>')+'<button class="text-button" data-note-bookmark="'+esc(note.id)+'" '+(!canMutate()?"disabled":"")+'>'+esc(note.bookmarked?"Bookmarked ★":"Bookmark ☆")+'</button>'+(note.question?'<button class="text-button" data-note-approve="'+esc(note.id)+'" '+(!canMutate() || note.review_status==="approved"?"disabled":"")+'>'+esc(note.review_status==="approved"?"Reviewed · approved":"Approve instruction draft")+'</button>':"")+'</div></article>').join("") : '<div class="empty-panel"><h3>No matching notes</h3><p>Change the search or bookmark filter. Source-linked notes appear as researchers save them.</p></div>';
  }
  function inspectCheckpoint(id) {
    const checkpoint=state.checkpoints.find(item=>String(item.id)===String(id));if(!checkpoint) return;
    const independent=checkpoint.checks?.independent_evaluation_passed===true;
    const chatSftRequired=needsChatSft(checkpoint), measured=!!checkpoint.calibration?.passed, checks=checkpoint.calibration?.checks || checkpoint.checks || {}, url=safeUrl(checkpoint.repo_id?"https://huggingface.co/"+checkpoint.repo_id:"");
    $("checkpoint-dialog-content").innerHTML='<h2>'+esc(checkpoint.name || checkpoint.repo_id || checkpoint.id)+'</h2><p class="record-summary">'+esc(checkpoint.baseline?"The existing chamber baseline remains unchanged. This observatory preserves it as the control.":mode==="preview"?"This is a placeholder from a simulated validation flow. No trained weights or measured calibration exists.":"A candidate remains distinct from the chamber until the owner selects a validated revision.")+'</p><div class="record-grid"><div><dt>Model / adapter</dt><dd>'+esc(checkpoint.base_model || checkpoint.model_id || "Not recorded")+'</dd></div><div><dt>Revision</dt><dd>'+esc(checkpoint.revision || checkpoint.artifact_revision || "No weights created")+'</dd></div><div><dt>Training job</dt><dd>'+esc(checkpoint.run_id || "Existing baseline")+'</dd></div><div><dt>Calibration</dt><dd><span class="tag '+(measured?"eligible":"review")+'">'+esc(checkpoint.baseline?"Retained baseline":measured?"Measured · passed":"Calibration required")+'</span></dd></div></div>'+(url?'<a class="source-external" href="'+esc(url)+'" target="_blank" rel="noopener noreferrer">Open model repository ↗</a>':"")+'<div class="record-block"><h3>Calibration record</h3><pre class="manifest-json">'+esc(JSON.stringify(checks,null,2))+'</pre><p class="rights-explanation">'+esc(mode==="preview"?"No test was run. The displayed requirements are examples, not measurements.":"The worker must measure compatibility, intervention vectors and control tasks for this exact model revision.")+'</p></div>'+(!checkpoint.baseline?'<div class="record-block"><h3>Independent evaluations</h3><p class="rights-explanation">'+esc(independent?'The sealed domain/general evaluation gate passed. Review the job record and suite limitations before interpreting the result.':'The candidate still needs the sealed domain/general evaluation gate. A calibration pass alone cannot unlock selection.')+'</p></div>':'')+(!checkpoint.baseline?'<div class="record-block"><h3>Manual selection</h3><p class="rights-explanation">Selecting a validated checkpoint records an operator choice. It does not reload the remote chamber; the deployment bridge must apply the pinned adapter and preserve the original baseline.</p><button class="primary-button" data-promote="'+esc(checkpoint.id)+'" '+(!measured || !independent || !checkpoint.checks?.passed || !canMutate() || chatSftRequired?"disabled":"")+'>'+esc(chatSftRequired?"SFT required for chamber chat":mode==="preview"?"Simulate selection proposal":"Select validated checkpoint")+'</button>'+(chatSftRequired?'<p class="method-note">This Llama Base checkpoint supports raw completion. Complete and calibrate a separate SFT stage before selecting it for chamber chat.</p>':!measured?'<p class="method-note">Selection is locked until measured calibration passes.</p>':"")+'</div>':"");
    if(checkpoint.deployment_env && Object.keys(checkpoint.deployment_env).length) {
      $("checkpoint-dialog-content").insertAdjacentHTML("beforeend",'<div class="record-block"><h3>Worker configuration</h3><p class="rights-explanation">Applies on operator redeployment. These pinned base, adapter and tokenizer settings do not reload the chamber automatically. No credentials are included.</p><pre class="manifest-json">'+esc(deploymentEnvironment(checkpoint))+'</pre><div class="dialog-actions"><button class="quiet-button" data-deployment-copy="'+esc(checkpoint.id)+'">Copy configuration</button><button class="quiet-button" data-deployment-download="'+esc(checkpoint.id)+'">Download .env</button></div></div>');
    }
    openDialog("checkpoint-dialog");
  }
  function deploymentEnvironment(checkpoint) {
    return Object.entries(checkpoint.deployment_env || {}).filter(([key,value])=>/^[A-Z_][A-Z0-9_]*$/.test(key) && ["string","number","boolean"].includes(typeof value)).map(([key,value])=>key+"="+JSON.stringify(String(value))).join("\n");
  }
  async function exportDeployment(id,download=false) {
    const checkpoint=state.checkpoints.find(item=>String(item.id)===String(id));if(!checkpoint)return;
    const content=deploymentEnvironment(checkpoint);if(!content){toast("No worker deployment configuration is available.");return;}
    if(!download) {
      try {await navigator.clipboard.writeText(content);toast("Worker configuration copied. Operator redeployment is still required.");}
      catch(_){toast("Clipboard is unavailable. Use Download .env or copy the displayed configuration.");}
      return;
    }
    const url=URL.createObjectURL(new Blob(["# Worker configuration; apply on operator redeployment.\n"+content+"\n"],{type:"text/plain"})), anchor=document.createElement("a");
    anchor.href=url;anchor.download=String(checkpoint.id).replace(/[^a-zA-Z0-9_-]/g,"_")+"-worker.env";document.body.appendChild(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    toast("Worker configuration downloaded. No chamber reload was requested.");
  }
  function confirmAction(title,message,label,callback) {
    text("confirm-title",title);text("confirm-message",message);text("confirm-accept",label);confirmation=callback;openDialog("confirm-dialog");
  }
  async function mutate(path,payload={},method="POST") {
    if(!canMutate()){explainOwner();return null;}
    busy=true;renderMission();renderDatasets();renderTraining();
    try { const result=await request(path,method,payload);await refreshState(true);return result; }
    catch(error){toast(error.message);return null;}
    finally {busy=false;render();}
  }
  function updatePreviewTimer(delay=150) {
    clearTimeout(previewTimer);previewTimer=null;
    if(mode!=="preview" || state.mission.status!=="running")return;
    previewTimer=setTimeout(()=>{
      previewTimer=null;
      if(mode!=="preview" || state.mission.status!=="running")return;
      if(document.hidden || currentView!=="research" || previewScrollFrame!==null || window.ObservatoryMotion?.isBusy()) {
        previewIdleAt=null;updatePreviewTimer();return;
      }
      const now=performance.now();
      if(previewIdleAt===null)previewIdleAt=now;
      if(now-previewIdleAt<(motionReduced()?2400:450)){updatePreviewTimer();return;}
      previewIdleAt=null;window.ObservatoryPreview.step(state,selectedAgent);render();updatePreviewTimer();
    },delay);
  }
  async function missionAction(action) {
    if(mode==="preview"){window.ObservatoryPreview.mission(state,action);if(action==="start")window.ObservatoryPreview.step(state,selectedAgent);previewIdleAt=null;updatePreviewTimer();render();toast("Preview "+action+" simulated. No real browser was started.");}
    else if(!canMutate()) explainOwner();
    else {const result=await mutate("admin/missions/"+action,action==="start"?{objective:preferences.objective || state.mission.objective}:{});if(result)toast("Mission "+action+" requested. The worker will report its actual state.");}
  }
  function reviewCurationRecovery(reviewId) {
    if(!curationRecoveryAvailable(reviewId)){toast("Recovery requires owner access and a paused, faulted or stopped mission.");return;}
    confirmAction("Authorize another curation attempt?","After any pending payment is reconciled, this authorizes another review attempt. The interrupted attempt may already have been billed. Saved completed reviews are preserved.","Acknowledge recovery",async()=>{
      if(!curationRecoveryAvailable(reviewId)){toast("The review or mission state changed. Refresh the recovery record.");return;}
      const result=await mutate("admin/curation/recovery",{reviewed:true,review_id:reviewId});
      if(result)toast("Recovery acknowledged. Completed reviews are preserved; resume research when ready.");
    });
  }
  function installExtraSettings() {
    $("setting-hf").closest("label").insertAdjacentHTML("beforebegin",'<div class="setup-divider"><h3>Original-document curation</h3><span>Opt-in model review</span></div><label class="checkbox-label"><input id="setting-auto-curation" type="checkbox"><span>Enable automated original-document curation<small>Two blind review passes classify consciousness research by research area and basis of claims. Scientific findings, philosophy and religious interpretations stay distinct. Verified rights, passed extraction and existing exclusions remain required. Uncertain cases await review. Runs during the research mission; uses research model billing. Q&amp;A approval remains separate.</small></span></label><p id="auto-curation-policy-notice" class="settings-note" role="status" hidden></p>');
    $("setting-browser").innerHTML='<option value="browseruse">Browser Use Cloud</option><option value="local">Local Chromium</option><option value="cdp">Existing CDP · including Steel</option>';
    const divider=$("setting-browser").closest("label");
    divider.insertAdjacentHTML("afterend",'<label id="browser-key-label">Browser Use API key<input id="setting-browser-key" type="password" autocomplete="off" placeholder="Leave blank to preserve the backend key"><small>Sent only to the authenticated backend. Never saved locally.</small></label><label id="cdp-label" hidden>CDP connection URL<input id="setting-cdp" type="password" autocomplete="off" spellcheck="false" placeholder="Private connection endpoint · not stored locally"></label><label id="chromium-label" hidden>Local Chromium executable<input id="setting-chromium" type="text" spellcheck="false" placeholder="Optional · backend filesystem path"></label>');
    $("cdp-label").insertAdjacentHTML("afterend",'<label id="cdp-ack-label" hidden><input id="setting-cdp-ack" type="checkbox"> I confirm this is a dedicated, isolated browser with no authenticated sessions.<small>Custom CDP runs one research agent. Do not attach a personal browser or expose its connection URL publicly.</small></label>');
    $("setting-provider").closest(".form-pair").insertAdjacentHTML("afterend",'<label id="research-key-label">Research provider API key<input id="setting-research-key" type="password" autocomplete="off" placeholder="Leave blank to preserve the backend key"><small>Stored encrypted by the backend; never in browser storage.</small></label><div class="form-pair"><label>Reasoning effort<select id="setting-effort"><option value="high">High</option><option value="xhigh">Extra high</option><option value="medium">Medium</option><option value="low">Low</option></select></label><label>Research agents<input id="setting-agent-count" type="number" min="1" max="6" value="6"></label></div>');
    $("setting-hf").closest("label").insertAdjacentHTML("afterend",'<label>Hugging Face token<input id="setting-hf-token" type="password" autocomplete="off" placeholder="Leave blank to preserve the backend token"></label><div class="setup-divider"><h3>Training permissions</h3><span>Explicit owner choices</span></div><label>Selected training base<input id="setting-base-model" type="text" spellcheck="false" value="meta-llama/Llama-3.1-70B"><small>Continued pretraining uses original text; instruction tuning is a separately approved SFT stage. Selecting a model does not download weights.</small></label><label><input id="setting-training-enabled" type="checkbox"> Enable actual training jobs<small>Training can allocate paid GPU jobs after backend validation. Preview actions never allocate compute.</small></label><label><input id="setting-synthetic-approved" type="checkbox"> Enable approved instruction examples<small>Generated notebook drafts require review. The owner must verify teacher-provider terms before permitting SFT.</small></label><label>Teacher-provider policy reference<input id="setting-policy-reference" type="text" placeholder="Policy URL / agreement and review date"><small>Required when enabling generated instruction examples; it does not bypass evidence or source-rights review.</small></label>');
    $("setting-policy-reference").closest("label").insertAdjacentHTML("afterend",'<details class="advanced-setup"><summary>Independent evaluation gates</summary><p>The bundled frozen suite is an authored integration smoke test. Supply an independently reviewed suite for scientific claims. Both the unchanged base and incoming adapter are compared with the candidate.</p><label>Frozen suite path<input id="setting-eval-suite" type="text" spellcheck="false" placeholder="Optional · absolute backend JSON path"></label><label>Expected suite SHA256<input id="setting-eval-sha" type="text" maxlength="64" spellcheck="false" placeholder="Required for a custom suite · pinned SHA256"></label><div class="form-pair"><label>Allowed accuracy drop<input id="setting-eval-accuracy-drop" type="number" min="0" max="1" step="0.01" value="0"></label><label>Maximum evaluation loss ratio<input id="setting-eval-nll-ratio" type="number" min="0.1" max="2" step="0.01" value="1"></label></div><div class="form-pair"><label>Minimum domain accuracy<input id="setting-eval-domain-min" type="number" min="0" max="1" step="0.05" value="0.5"></label><label>Minimum general accuracy<input id="setting-eval-general-min" type="number" min="0" max="1" step="0.05" value="0.5"></label></div><div class="form-pair"><label>Evaluation sequence length<input id="setting-eval-max-length" type="number" min="64" max="32768" value="1024"></label><label>Maximum held-out loss ratio<input id="setting-holdout-ratio" type="number" min="0.1" max="2" step="0.01" value="1"></label></div></details>');
    all("#settings-form .settings-note").forEach(item=>item.textContent="Secret fields are sent only to the authenticated backend and are never written to browser storage. Public settings report configured status, not keys.");
    $("setting-policy-reference").closest("label").insertAdjacentHTML("afterend",'<details class="advanced-setup"><summary>GPU worker, repositories &amp; recipe</summary><p>These settings bound training jobs. Research browsing continues until manually stopped.</p><label>Dataset repository<input id="setting-dataset-repo" type="text" spellcheck="false" placeholder="namespace/consciousness-corpus"></label><label>Model repository<input id="setting-model-repo" type="text" spellcheck="false" placeholder="namespace/Llama-Consciousness-70B"></label><label>Pinned worker container<input id="setting-training-image" type="text" spellcheck="false" placeholder="registry/worker@sha256:…"></label><div class="form-pair"><label>GPU hardware<select id="setting-hardware"><option value="a100-large">A100 large</option><option value="h200">H200</option></select></label><label>Adaptation<select id="setting-training-mode"><option value="qlora">QLoRA</option><option value="lora">LoRA</option></select></label></div><div class="form-pair"><label>Check snapshots every (hours)<input id="setting-training-interval" type="number" min="0.1" step="0.1" value="4"></label><label>GPU timeout (seconds)<input id="setting-training-timeout" type="number" min="60" max="604800" value="14400"></label></div><div class="form-pair"><label>Minimum documents<input id="setting-min-documents" type="number" min="1" value="20"></label><label>Minimum tokens<input id="setting-min-tokens" type="number" min="1" value="50000"></label></div><div class="form-pair"><label>Training steps per job<input id="setting-max-steps" type="number" min="1" value="100"></label><label>Sequence length<input id="setting-sequence-length" type="number" min="128" value="2048"></label></div><label>Model publication<select id="setting-publish-policy"><option value="private">Private repositories</option><option value="hold">Hold publication</option><option value="public">Public model artifacts</option></select></label><label>Budget reference (USD)<input id="setting-training-budget" type="number" min="0" step="1" placeholder="Optional"><small>Informational. Provider funds, hardware choices and job timeouts control spending; this field is not an enforced spending cap.</small></label><label>Base model revision<input id="setting-base-revision" type="text" spellcheck="false" placeholder="Blank resolves and pins the selected repository at preparation"></label></details>');
    $("setting-dataset-repo").closest("label").insertAdjacentHTML("beforebegin",'<label class="checkbox-label"><input id="setting-continue-training" type="checkbox" checked><span>Continue from the previous validated pretraining adapter<small>Use the latest validated CPT adapter as the parent for a new curated snapshot. Disable to start from the pinned original base. Failed jobs are never retried automatically.</small></span></label>');
    $("setting-training-enabled").closest("label").querySelector("small").textContent="When enabled, the backend checks new snapshots on the configured interval and can allocate paid GPU jobs. Preview never allocates compute.";
    $("setting-browser").addEventListener("change",toggleBrowserFields);
    $("setting-base-model").addEventListener("input",clearInheritedBaseRevision);
    $("setting-base-revision").addEventListener("input",()=>{inheritedBaseRevision=false;});
  }
  function toggleBrowserFields() {
    const value=$("setting-browser").value;
    $("browser-key-label").hidden=value!=="browseruse";$("cdp-label").hidden=value!=="cdp";$("chromium-label").hidden=value!=="local";
    $("cdp-ack-label").hidden=value!=="cdp";$("setting-agent-count").disabled=value==="cdp";
    if(value==="cdp") $("setting-agent-count").value=1;
  }
  function clearInheritedBaseRevision() {
    if(inheritedBaseRevision && $("setting-base-model").value.trim()!==defaults.hf_base_model) {
      $("setting-base-revision").value="";
      inheritedBaseRevision=false;
    }
  }
  function fillExtraSettings() {
    const settings=mergeSettings(preferences,state.settings);
    $("setting-effort").value=settings.reasoning_effort || "high";$("setting-agent-count").value=settings.agent_count || 6;
    $("setting-base-model").value=settings.hf_base_model || defaults.hf_base_model;
    $("setting-training-enabled").checked=!!settings.training_enabled;$("setting-synthetic-approved").checked=!!settings.synthetic_training_approved;
    fillAutoCurationSettings(settings);
    $("setting-continue-training").checked=settings.training_continue_from_previous!==false;
    $("setting-policy-reference").value=settings.provider_policy_reference || "";$("setting-chromium").value=settings.chromium_executable || "";
    $("setting-cdp-ack").checked=!!settings.cdp_isolated_ack;
    $("setting-protocol").value=settings.research_protocol || "responses";
    const fields={"dataset-repo":["hf_dataset_repo",""],"model-repo":["hf_model_repo",""],"training-image":["training_image",""],hardware:["training_hardware","a100-large"],"training-mode":["training_mode","qlora"],"training-interval":["training_interval_hours",4],"training-timeout":["training_timeout_seconds",14400],"min-documents":["training_min_documents",20],"min-tokens":["training_min_tokens",50000],"max-steps":["training_max_steps",100],"sequence-length":["training_sequence_length",2048],"publish-policy":["publish_policy","private"],"training-budget":["training_budget_usd",""],"base-revision":["hf_base_revision",""]};
    Object.entries(fields).forEach(([field,[key,fallback]])=>{$("setting-"+field).value=settings[key] ?? fallback;});
    const evalFields={"eval-suite":["training_eval_suite_path",""],"eval-sha":["training_eval_expected_sha256",""],"eval-accuracy-drop":["training_eval_max_accuracy_drop",0],"eval-nll-ratio":["training_eval_max_nll_ratio",1],"eval-domain-min":["training_eval_min_domain_accuracy",0.5],"eval-general-min":["training_eval_min_general_accuracy",0.5],"eval-max-length":["training_eval_max_length",1024],"holdout-ratio":["training_max_loss_ratio",1]};
    Object.entries(evalFields).forEach(([field,[key,fallback]])=>{$("setting-"+field).value=settings[key] ?? fallback;});
    inheritedBaseRevision=settings.hf_base_model===defaults.hf_base_model && $("setting-base-revision").value===defaults.hf_base_revision;
    toggleBrowserFields();
    syncResearchFields(settings.research_model);
  }
  function fillAutoCurationSettings(settings) {
    const needsUpdate=settings.auto_curation_enabled===true && settings.auto_curation_policy_ack!==autoCurationPolicyAck;
    $("setting-auto-curation").checked=settings.auto_curation_enabled===true && !needsUpdate;
    $("auto-curation-policy-notice").hidden=!needsUpdate;
    $("auto-curation-policy-notice").textContent=needsUpdate?'Your saved review policy predates the broader consciousness classification. Automated review remains idle. Enable the updated policy above and save setup to authorize it.':'';
  }
  function collectSettings() {
    clearInheritedBaseRevision();
    const settings={objective:$("setting-objective").value.trim(),research_provider:$("setting-provider").value,research_model:$("setting-model").value.trim(),reasoning_effort:$("setting-effort").value,agent_count:Math.max(1,Math.min(6,Number($("setting-agent-count").value) || 6)),browser_provider:$("setting-browser").value,hf_namespace:$("setting-hf").value.trim(),hf_base_model:$("setting-base-model").value.trim(),training_enabled:$("setting-training-enabled").checked,training_continue_from_previous:$("setting-continue-training").checked,synthetic_training_approved:$("setting-synthetic-approved").checked,provider_policy_reference:$("setting-policy-reference").value.trim(),chromium_executable:$("setting-chromium").value.trim()};
    settings.research_protocol=$("setting-protocol").value;
    settings.auto_curation_enabled=$("setting-auto-curation").checked;
    settings.auto_curation_policy_ack=settings.auto_curation_enabled?autoCurationPolicyAck:"";
    if(settings.research_provider==="x402") {
      settings.research_model=$("setting-catalog-model").value;
      if(settings.research_model && !compatibleResearchModels(settings.research_protocol).some(item=>item.id===settings.research_model))throw new Error("Select a compatible model from the live broker catalog before changing x402 research setup.");
      if(!settings.research_model)delete settings.research_model;
    }
    if(settings.synthetic_training_approved && !settings.provider_policy_reference) throw new Error("Record the teacher-provider policy reference before enabling instruction examples.");
    if(settings.research_provider!=="x402" && !settings.research_model || !settings.hf_base_model) throw new Error("Research and subject model IDs must be provided.");
    settings.cdp_isolated_ack=$("setting-cdp-ack").checked;
    if(settings.browser_provider==="cdp") {
      settings.agent_count=1;
      if(!settings.cdp_isolated_ack) throw new Error("Confirm the custom CDP browser is isolated and contains no authenticated sessions.");
    }
    Object.assign(settings,{training_provider:"hf_jobs",activation_policy:"manual",hf_dataset_repo:$("setting-dataset-repo").value.trim(),hf_model_repo:$("setting-model-repo").value.trim(),training_image:$("setting-training-image").value.trim(),training_hardware:$("setting-hardware").value,training_mode:$("setting-training-mode").value,training_interval_hours:Number($("setting-training-interval").value),training_timeout_seconds:Number($("setting-training-timeout").value),training_min_documents:Number($("setting-min-documents").value),training_min_tokens:Number($("setting-min-tokens").value),training_max_steps:Number($("setting-max-steps").value),training_sequence_length:Number($("setting-sequence-length").value),publish_policy:$("setting-publish-policy").value,training_budget_usd:$("setting-training-budget").value===""?null:Number($("setting-training-budget").value),hf_base_revision:$("setting-base-revision").value.trim() || null});
    Object.assign(settings,{training_eval_suite_path:$("setting-eval-suite").value.trim(),training_eval_expected_sha256:$("setting-eval-sha").value.trim(),training_eval_max_accuracy_drop:Number($("setting-eval-accuracy-drop").value),training_eval_max_nll_ratio:Number($("setting-eval-nll-ratio").value),training_eval_min_domain_accuracy:Number($("setting-eval-domain-min").value),training_eval_min_general_accuracy:Number($("setting-eval-general-min").value),training_eval_max_length:Number($("setting-eval-max-length").value),training_max_loss_ratio:Number($("setting-holdout-ratio").value)});
    if(settings.training_eval_suite_path && !/^[a-f0-9]{64}$/.test(settings.training_eval_expected_sha256))throw new Error("A custom evaluation suite needs its canonical SHA256 fingerprint.");
    return settings;
  }
  async function saveSetup(event) {
    event.preventDefault();text("setup-result","");
    try {
      const nextMode=document.querySelector('input[name="mode"]:checked').value, api=validateEndpoint($("setting-api").value);
      // A connected setup must select from that backend's actual catalog, never
      // from the authored preview fixtures or a previous endpoint's catalog.
      if(nextMode==="connected" && (mode!=="connected" || api!==preferences.api_base)) {
        preferences.api_base=api;disconnect();clearTimeout(previewTimer);mode="connected";state=emptyState();selectedAgent="";lastPreviewSource="";
        await refreshState();
        if(!connected)return;
        syncResearchFields("");
      }
      if(nextMode==="preview" && mode!=="preview") {
        disconnect();mode="preview";state=window.ObservatoryPreview.create();selectedAgent="";lastPreviewSource="";syncResearchFields("");
      }
      const settings=collectSettings();
      ownerToken=$("owner-token").value.trim();preferences={...preferences,...settings,api_base:api};
      // Explicit allowlist: never serialize form data or include credential fields.
      const publicPreferences={};
      ["api_base","objective","research_provider","research_model","research_protocol","reasoning_effort","agent_count","browser_provider","cdp_isolated_ack","hf_namespace","hf_base_model","training_enabled","training_continue_from_previous","auto_curation_enabled","auto_curation_policy_ack","synthetic_training_approved","provider_policy_reference","chromium_executable","hf_dataset_repo","hf_model_repo","training_image","training_hardware","training_mode","training_interval_hours","training_timeout_seconds","training_min_documents","training_min_tokens","training_max_steps","training_sequence_length","publish_policy","training_budget_usd","hf_base_revision","training_eval_suite_path","training_eval_expected_sha256","training_eval_max_accuracy_drop","training_eval_max_nll_ratio","training_eval_min_domain_accuracy","training_eval_min_general_accuracy","training_eval_max_length","training_max_loss_ratio"].forEach(key=>publicPreferences[key]=preferences[key]);
      try {localStorage.setItem(storageKey,JSON.stringify(publicPreferences));}catch(_){}
      if(nextMode!==mode){disconnect();clearTimeout(previewTimer);mode=nextMode;state=mode==="preview"?window.ObservatoryPreview.create():emptyState();selectedAgent="";lastPreviewSource="";}
      if(mode==="preview") {
        state.settings={...state.settings,...settings};state.mission.objective=settings.objective;
        ["setting-research-key","setting-browser-key","setting-hf-token","setting-cdp"].forEach(id=>$(id).value="");
        render();
        text("setup-result","Preview preferences saved. Secrets were not saved or sent anywhere.");toast("Preview setup saved.");return;
      }
      disconnect();await refreshState();
      if(!connected){render();return;}
      if(ownerToken) {
        const secretSettings={...settings};
        const brainKey=$("setting-research-key").value.trim(), browserKey=$("setting-browser-key").value.trim(), hfToken=$("setting-hf-token").value.trim(), cdp=$("setting-cdp").value.trim();
        if(brainKey && settings.research_provider!=="x402") secretSettings[settings.research_provider==="anthropic"?"anthropic_api_key":"openai_api_key"]=brainKey;
        if(browserKey) secretSettings.browser_use_api_key=browserKey;if(hfToken) secretSettings.hf_token=hfToken;if(cdp)secretSettings.cdp_url=cdp;
        await request("admin/settings","POST",secretSettings);
        ["setting-research-key","setting-browser-key","setting-hf-token","setting-cdp"].forEach(id=>$(id).value="");
        await refreshState(true);text("setup-result","Connected setup saved by the backend. Secret input fields cleared.");toast("Connected operator setup saved.");
      } else {text("setup-result","Connected as a read-only observer. Enter the owner token to save backend settings.");toast("Connected as a read-only observer.");}
      render();
    } catch(error) {text("setup-result",error.message);toast(error.message);}
  }
  function collectSourceReview(data) {
    const verified=data.has("license_verified"), evidence=String(data.get("rights_evidence") || "").trim(), license=String(data.get("license") || "unknown").trim();
    if(verified && !evidence)throw new Error("Rights verification requires recorded evidence for the exact copy.");
    const payload={review_status:data.get("review_status"),license,license_verified:verified,rights_evidence:evidence,review_note:String(data.get("review_note") || "").trim(),rights_status:verified?"license_verified":data.get("review_status")==="quarantined"?"reference_only":"needs_review"};
    const rationale=String(data.get("quality_rationale") || "").trim();
    if(payload.review_status==="approved" && !rationale)throw new Error("Record a quality rationale before approving corpus inclusion.");
    if(rationale) {
      const domains=topicDomains.map(([id])=>id).filter(id=>data.has('topic_'+id)), kind=data.get('evidence_kind'), stance=data.get('evidence_stance');
      if(!domains.length)throw new Error("Select at least one research area for this document.");
      if(!evidenceKinds.some(([id])=>id===kind))throw new Error("Choose the basis of this document's claims.");
      if(!domains.includes('machine_consciousness') && stance!=='not_applicable')throw new Error("Use Not applicable when this document does not address machine consciousness.");
      const covered=['mixed','methodological'].includes(stance)?['supportive','skeptical','uncertain'].filter(item=>data.has('covered_'+item)):[];
      payload.quality_review={status:payload.review_status==="approved"?"approved":payload.review_status==="rejected"?"rejected":"pending",reviewed_by:String(data.get("quality_reviewed_by") || "").trim(),rationale,topic_relevance:data.get("topic_relevance"),topic_domains:domains,evidence_kind:kind,evidence_stance:stance,source_type:data.get("quality_source_type"),covered_stances:covered};
    }
    payload.contains_benchmark=data.has("contains_benchmark");payload.chamber_stimulus=data.has("chamber_stimulus");
    payload.extraction_review_status=data.has("extraction_review_approved")?"approved":"pending";
    payload.extraction_review_evidence=String(data.get("extraction_review_evidence") || "").trim();
    return payload;
  }
  async function reviewSource(form) {
    const id=form.dataset.reviewSource, source=sourceById(id);
    if(!source || !canMutate()) return;
    let payload;
    try {payload=collectSourceReview(new FormData(form));}catch(error){toast(error.message);return;}
    const verified=payload.license_verified, evidence=payload.rights_evidence, license=payload.license;
    if(mode==="preview") {
      Object.assign(source,payload);
      const compatible=/^(cc-by(?:-4\.0|-3\.0)?|cc0(?:-1\.0)?|public-domain)$/i.test(license);
      const allowed=verified && !!evidence && compatible && payload.review_status==="approved" && payload.quality_review?.topic_relevance==="relevant" && !payload.contains_benchmark && !payload.chamber_stimulus && (source.extraction?.quality==="passed" || payload.extraction_review_status==="approved" && !!payload.extraction_review_evidence);
      source.curation={eligible:allowed,status:allowed?"eligible":"quarantined",reasons:allowed?[]:[!compatible?"license_not_in_public_corpus_policy":!verified?"rights_not_verified":"owner_review_excludes_source"]};
      window.ObservatoryPreview.event(state,"source.reviewed","Simulated owner review saved. No actual rights were granted.",source.agent_id,{source_id:id});render();inspectSource(id);toast("Preview review saved.");
    } else {const result=await mutate("admin/sources/"+encodeURIComponent(id),payload,"PATCH");if(result){inspectSource(id);toast("Source review saved; eligibility was evaluated by the backend.");}}
  }
  async function saveNote(event) {
    event.preventDefault();const value=$("note-text").value.trim(), source=currentSource();if(!value || !source)return;
    const payload={text:value,source_id:source.id,agent_id:selectedAgent,type:"owner_note"};
    if(mode==="preview") {
      const note={...payload,id:"preview-owner-note-"+Date.now(),generated_by:"owner",review_status:"pending",bookmarked:false,created_at:new Date().toISOString()};
      state.notes.push(note);window.ObservatoryPreview.event(state,"note.saved","Simulated owner note saved.",selectedAgent,{note_id:note.id,source_id:source.id});$("note-text").value="";notebook="notes";render();toast("Preview note saved.");
    } else {const result=await mutate("admin/notes",payload);if(result){$("note-text").value="";notebook="notes";renderNotebook();toast("Source-linked note saved.");}}
  }
  async function editNote(id,payload) {
    const note=state.notes.find(item=>String(item.id)===String(id));if(!note)return;
    if(mode==="preview"){Object.assign(note,payload);renderAllNotes();renderNotebook();toast("Preview note updated.");}
    else {const result=await mutate("admin/notes/"+encodeURIComponent(id),payload,"PATCH");if(result){renderAllNotes();toast("Note updated.");}}
  }
  function downloadManifest() {
    const dataset=state.datasets.find(item=>String(item.id)===String(selectedDataset));if(!dataset)return;
    const payload={id:dataset.id,created_at:dataset.created_at,manifest_hash:dataset.manifest_hash,counts:dataset.counts,manifest:dataset.manifest,notice:mode==="preview"?"Simulated metadata manifest. No training documents or weights.":"Public metadata manifest; original documents remain in the private dataset."};
    const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"})), anchor=document.createElement("a");
    anchor.href=url;anchor.download=String(dataset.id).replace(/[^a-zA-Z0-9_-]/g,"_")+"-manifest.json";document.body.appendChild(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    toast(mode==="preview"?"Preview metadata manifest downloaded.":"Public metadata manifest downloaded.");
  }
  async function downloadDataset() {
    const dataset=state.datasets.find(item=>String(item.id)===String(selectedDataset));
    if(!dataset || mode==="preview"){toast("Preview contains metadata only. Sealed dataset exports require an actual backend snapshot.");return;}
    if(!canMutate()){explainOwner();return;}
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),60000);
    $("dataset-export").disabled=true;
    try {
      const response=await fetch(endpoint("admin/datasets/"+encodeURIComponent(dataset.id)+"/export"),{headers:{Accept:"application/zip",Authorization:"Bearer "+ownerToken},credentials:"omit",cache:"no-store",signal:controller.signal});
      if(!response.ok){const result=await response.json().catch(()=>null);throw new Error(typeof result?.detail==="string"?result.detail:"Dataset export returned HTTP "+response.status+".");}
      if(!(response.headers.get("Content-Type") || "").startsWith("application/zip"))throw new Error("The backend did not return a dataset ZIP.");
      const blob=await response.blob(),url=URL.createObjectURL(blob),anchor=document.createElement("a");
      anchor.href=url;anchor.download=String(dataset.id).replace(/[^a-zA-Z0-9_-]/g,"_")+".zip";document.body.appendChild(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
      toast("Sealed snapshot downloaded with separate CPT/SFT splits, provenance and file hashes.");
    } catch(error){toast(error.name==="AbortError"?"Dataset export timed out. No paid job or upload was requested.":error.message);}
    finally {clearTimeout(timeout);renderDatasets();}
  }
  async function createSnapshot() {
    if(mode==="preview"){const dataset=window.ObservatoryPreview.snapshot(state);selectedDataset=dataset.id;render();toast("Simulated metadata snapshot created. No corpus was uploaded.");}
    else {const result=await mutate("admin/snapshots",{});if(result){selectedDataset=result.id;renderDatasets();renderTraining();toast("Immutable corpus candidate created.");}}
  }
  function startTraining() {
    const snapshot=$("training-dataset").value, stage=$("training-stage").value, parent=$("training-parent").value;
    if(!snapshot)return;
    confirmAction(mode==="preview"?"Validate a preview job?":"Submit a training job?",mode==="preview"?"This demonstrates validation and creates a placeholder checkpoint. It does not allocate compute, train weights, upload documents or measure calibration.":"The authenticated backend will validate the sealed dataset and submit a paid GPU job if training is enabled. The existing chamber baseline remains preserved.",mode==="preview"?"Validate preview":"Submit job",async()=>{
      if(mode==="preview"){const job=window.ObservatoryPreview.train(state,snapshot,stage,parent || null);selectedJob=job.id;render();toast("Preview job validated. No training ran.");}
      else {const payload=stage==="sft"?{stage:"sft",parent_run_id:parent,snapshot_id:snapshot}:{snapshot_id:snapshot};const result=await mutate("admin/train",payload);if(result){if(result.status==="not_ready"){toast("Training not submitted: "+list(result.reasons).map(titleCase).join("; "));return;}if(result.id)selectedJob=result.id;renderTraining();toast(/failed|unknown/.test(result.status || "")?"Submission needs inspection: "+titleCase(result.status):"Training submission recorded. Watch the worker status.");}}
    });
  }
  function retryTraining(id) {
    const job=state.jobs.find(item=>String(item.id)===String(id));
    if(mode!=="connected" || !job || !["failed","cancelled"].includes(job.status))return;
    if(!canMutate()){explainOwner();return;}
    const snapshot=job.snapshot_id || job.manifest?.snapshot_id;
    if(!snapshot){toast("This job has no recorded immutable snapshot to retry.");return;}
    const payload={snapshot_id:snapshot,retry:true};
    if(job.stage==="sft"){payload.stage="sft";payload.parent_run_id=job.parent_run_id || job.manifest?.parent_run_id;if(!payload.parent_run_id){toast("The SFT job has no recorded parent checkpoint.");return;}}
    confirmAction("Retry the "+titleCase(job.status).toLowerCase()+" job?","This is a new paid GPU submission for the recorded immutable snapshot. The backend validates current settings and records its link to the previous job. Retries require this explicit owner action; they are never scheduled automatically.","Retry job",async()=>{
      const result=await mutate("admin/train",payload);
      if(!result)return;
      if(result.status==="not_ready"){toast("Retry not submitted: "+list(result.reasons).map(titleCase).join("; "));return;}
      if(result.id)selectedJob=result.id;renderTraining();
      toast(/failed|unknown/.test(result.status || "")?"Retry needs inspection: "+titleCase(result.status):"Manual retry recorded. Watch the worker status.");
    });
  }
  document.addEventListener("click",async event=>{
    const curationRecovery=event.target.closest("[data-curation-recover]");if(curationRecovery){reviewCurationRecovery(curationRecovery.dataset.curationRecover);return;}
    const view=event.target.closest("[data-view]");if(view){showView(view.dataset.view);return;}
    const agentButton=event.target.closest("[data-agent]");if(agentButton){selectedAgent=agentButton.dataset.agent;clearLiveFrame();lastPreviewSource="";renderAgents();renderBrowser();renderNotebook();renderEvents();return;}
    const notebookButton=event.target.closest("[data-notebook]");if(notebookButton){notebook=notebookButton.dataset.notebook;renderNotebook();return;}
    const sourceButton=event.target.closest("[data-source]");if(sourceButton){inspectSource(sourceButton.dataset.source);return;}
    const datasetButton=event.target.closest("[data-dataset]");if(datasetButton){selectedDataset=datasetButton.dataset.dataset;renderDatasets();renderTraining();return;}
    const manifestButton=event.target.closest("[data-manifest]");if(manifestButton){manifestView=manifestButton.dataset.manifest;renderDatasets();return;}
    const checkpointButton=event.target.closest("[data-checkpoint]");if(checkpointButton){inspectCheckpoint(checkpointButton.dataset.checkpoint);return;}
    const deploymentCopy=event.target.closest("[data-deployment-copy]");if(deploymentCopy){await exportDeployment(deploymentCopy.dataset.deploymentCopy);return;}
    const deploymentDownload=event.target.closest("[data-deployment-download]");if(deploymentDownload){await exportDeployment(deploymentDownload.dataset.deploymentDownload,true);return;}
    const closeButton=event.target.closest("[data-close-dialog]");if(closeButton){closeButton.closest("dialog").close();return;}
    const bookmark=event.target.closest("[data-bookmark-source]");if(bookmark){bookmarkSource(bookmark.dataset.bookmarkSource);inspectSource(bookmark.dataset.bookmarkSource);return;}
    const noteBookmark=event.target.closest("[data-note-bookmark]");if(noteBookmark){const note=state.notes.find(item=>String(item.id)===noteBookmark.dataset.noteBookmark);if(note)await editNote(note.id,{bookmarked:!note.bookmarked});return;}
    const approve=event.target.closest("[data-note-approve]");if(approve){await editNote(approve.dataset.noteApprove,{review_status:"approved"});return;}
    const retry=event.target.closest("[data-retry-job]");if(retry){retryTraining(retry.dataset.retryJob);return;}
    const promote=event.target.closest("[data-promote]");if(promote){confirmAction("Select the validated checkpoint?","This records an operator selection of a pinned checkpoint. The remote chamber is not reloaded by this request; its deployment bridge remains a separate action.","Record selection",async()=>{const result=await mutate("admin/checkpoints/"+encodeURIComponent(promote.dataset.promote)+"/activate",{});if(result){inspectCheckpoint(promote.dataset.promote);toast("Validated checkpoint selected. Remote chamber reload remains pending.");}});return;}
    const cancel=event.target.closest("[data-cancel-job]");if(cancel){confirmAction("Cancel the GPU job?","Cancellation is sent to the provider. Partial logs and the immutable dataset remain in the record.","Cancel job",()=>mutate("admin/train/"+encodeURIComponent(cancel.dataset.cancelJob)+"/cancel",{}));}
  });
  document.addEventListener("submit",event=>{const form=event.target.closest("[data-review-source]");if(form){event.preventDefault();reviewSource(form);}});
  document.addEventListener("change",event=>{const form=event.target.closest("[data-review-source]"), domainChanged=topicDomains.some(([id])=>event.target.name==='topic_'+id);if(form && (event.target.name==='evidence_stance' || domainChanged))syncSourceReviewFields(form,domainChanged);});
  all("dialog").forEach(dialog=>{dialog.addEventListener("click",event=>{if(event.target===dialog){const box=dialog.getBoundingClientRect();if(event.clientX<box.left || event.clientX>box.right || event.clientY<box.top || event.clientY>box.bottom)dialog.close();}});});
  all("[data-view]").forEach(button=>button.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;event.preventDefault();const buttons=all("[data-view]"),index=buttons.indexOf(button),next=event.key==="Home"?0:event.key==="End"?buttons.length-1:(index+(event.key==="ArrowRight"?1:-1)+buttons.length)%buttons.length;showView(buttons[next].dataset.view,true);}));
  $("setup-open").addEventListener("click",openSetup);$("connect-open").addEventListener("click",openSetup);
  $("settings-form").addEventListener("submit",saveSetup);$("note-form").addEventListener("submit",saveNote);
  $("owner-token").addEventListener("input",()=>{ownerToken=$("owner-token").value.trim();renderNotebook();renderTraining();renderDatasets();});
  $("setting-provider").addEventListener("change",()=>{const next=$("setting-provider").value,old=$("setting-model").value;if(next!=="x402" && (previousProvider==="x402" || ["gpt-6-astra","claude-fable-5-1"].includes(old)))$("setting-model").value=next==="anthropic"?"claude-fable-5-1":"gpt-6-astra";$("setting-research-key").value="";previousProvider=next;syncResearchFields(next==="x402"?"":undefined);});
  $("setting-protocol").addEventListener("change",()=>syncResearchFields(""));
  $("setting-catalog-model").addEventListener("change",()=>{const model=compatibleResearchModels($("setting-protocol").value).find(item=>item.id===$("setting-catalog-model").value);text("research-catalog-status",mode==="preview"?"Simulated selection. No model call or payment occurs.":(model?.capability_source==="owner_declared"?"Owner-declared controls.":"Gateway-advertised controls.")+" Paid compatibility remains untested. Validate the chosen model with a bounded acceptance test before continuous research.");});
  $("connection-check").addEventListener("click",async()=>{try{preferences.api_base=validateEndpoint($("setting-api").value);const payload=normalizeState(await request("state"));text("setup-result","Backend reachable. "+payload.agents.length+" actual agents, "+payload.sources.length+" source records. Save setup to enter connected mode.");if(mode==="connected"){state=payload;connected=true;connectionError="";render();}}catch(error){text("setup-result",error.message+" Preview mode was not substituted.");}});
  $("preview-reset").addEventListener("click",()=>{if(mode!=="preview"){toast("Switch to Preview to reset simulated records.");return;}clearTimeout(previewTimer);state=window.ObservatoryPreview.create();state.settings=mergeSettings(state.settings,preferences);if(preferences.objective)state.mission.objective=preferences.objective;selectedAgent="";selectedDataset="";selectedJob="";lastPreviewSource="";render();toast("Preview reset. No connected backend was changed.");});
  $("mission-toggle").addEventListener("click",()=>missionAction(state.mission.status==="running"?"pause":["paused","funding_paused","faulted"].includes(state.mission.status)?"resume":"start"));
  $("mission-stop").addEventListener("click",()=>confirmAction(mode==="preview"?"Stop the preview?":"Stop the research mission?",mode==="preview"?"The simulation stops. Example notes, sources and snapshots remain available.":"The worker stops research and releases its owned browser sessions. Source records, notes and datasets remain durable.",mode==="preview"?"Stop preview":"Stop mission",()=>missionAction("stop")));
  $("mission-step").addEventListener("click",()=>{
    if(mode!=="preview")return;
    if(state.mission.status==="running" && (previewScrollFrame!==null || window.ObservatoryMotion?.isBusy())){toast("Let the current passage finish before advancing.");return;}
    previewIdleAt=null;window.ObservatoryPreview.step(state,selectedAgent);render();updatePreviewTimer();toast("Advanced one simulated research event.");
  });
  $("browser-inspect").addEventListener("click",()=>{const source=currentSource();if(source)inspectSource(source.id);});
  $("capture-inspect").addEventListener("click",()=>{const source=currentSource();if(source)inspectSource(source.id);});
  $("bookmark-source").addEventListener("click",()=>{const source=currentSource();if(source)bookmarkSource(source.id);});
  ["notes-open","show-notes"].forEach(id=>$(id).addEventListener("click",()=>{renderAllNotes();openDialog("notes-dialog");}));
  $("notes-search").addEventListener("input",renderAllNotes);$("notes-bookmarked").addEventListener("change",renderAllNotes);
  $("evidence-search").addEventListener("input",renderEvidence);$("rights-filter").addEventListener("change",renderEvidence);$("agent-filter").addEventListener("change",renderEvidence);
  $("event-filter-toggle").addEventListener("click",()=>{eventOnlyAgent=!eventOnlyAgent;renderEvents();});
  $("snapshot-create").addEventListener("click",createSnapshot);$("manifest-download").addEventListener("click",downloadManifest);
  $("dataset-export").addEventListener("click",downloadDataset);
  $("training-start").addEventListener("click",startTraining);$("training-dataset").addEventListener("change",renderTraining);$("training-stage").addEventListener("change",renderTraining);$("training-parent").addEventListener("change",renderTraining);
  $("training-job-select").addEventListener("change",()=>{selectedJob=$("training-job-select").value;renderTraining();});
  $("confirm-accept").addEventListener("click",async()=>{const callback=confirmation;confirmation=null;$("confirm-dialog").close();if(callback)await callback();});
  window.addEventListener("beforeunload",()=>{disconnect();clearTimeout(previewTimer);ownerToken="";});
  document.addEventListener("visibilitychange",()=>{if(document.hidden){cancelPreviewScroll();window.ObservatoryMotion?.stop();}else if(mode==="connected")refreshState(true);else renderPreviewViewport();});
  lockObserverViewport($("browser-display"));
  window.addEventListener("resize",()=>{if(mode==="preview")renderPreviewViewport();else renderFrameTelemetry();});
  installExtraSettings();$("setting-provider").value=preferences.research_provider;$("setting-browser").value=preferences.browser_provider;fillExtraSettings();render();showView(location.hash.slice(1) || "research");
  if(mode==="connected")refreshState();
  if(mode==="preview" && parameters.get("motion")==="1")missionAction("start");
  setInterval(refreshFrame,2000);
})();
