/* Real interface against authored API fixtures; no provider calls or paid jobs. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.OBSERVATORY_TEST_CHROMIUM || undefined});
  const context=await browser.newContext({viewport:{width:1440,height:1000}});
  await context.route('**/*',route=>new URL(route.request().url()).hostname==='127.0.0.1'?route.continue():route.abort());
  const page=await context.newPage(), errors=[], submissions=[], settingsRequests=[];
  page.on('pageerror',error=>errors.push(error.message));
  const base=process.env.OBSERVATORY_UI_URL || 'http://127.0.0.1:8060';
  // The ordinary public entry must try the real service, never fabricate a swarm.
  await page.goto(base+'/observatory.html');
  await page.waitForTimeout(400);
  assert.equal(await page.locator('[data-agent]').count(),0);
  assert.doesNotMatch(await page.locator('#mode-badge').innerText(),/Preview/);
  const fixture={mission:{id:'fixture',status:'stopped'},agents:[],sources:[],notes:[],events:[],cursor:0,
    datasets:[{id:'sealed-1',status:'sealed',document_count:25,token_count:90000,manifest:{snapshot_id:'sealed-1'}}],
    jobs:[{id:'failed-1',stage:'sft',parent_run_id:'cpt-1',snapshot_id:'sealed-1',status:'failed'}],
    settings:{synthetic_training_approved:true,provider_policy_reference:'fixture teacher agreement'},
    checkpoints:[{id:'selected-1',status:'selected',base_model:'fixture/base',revision:'b'.repeat(40),checks:{passed:true},calibration:{passed:true},deployment_env:{CHAMBER_MODEL:'fixture/base',MODEL_ADAPTER_ID:'fixture/adapter',MODEL_ADAPTER_REVISION:'b'.repeat(40),MODEL_ADAPTER_SUBFOLDER:'runs/job/adapter'}}]};
  await context.route('**/api/**',async route=>{
    const request=route.request(), pathname=new URL(request.url()).pathname;
    if(pathname==='/api/state')return route.fulfill({json:fixture});
    if(pathname==='/api/events')return route.fulfill({contentType:'text/event-stream',body:': fixture\n\n'});
    if(pathname==='/api/admin/settings'){
      settingsRequests.push({payload:request.postDataJSON(),authorization:request.headers().authorization});
      return route.fulfill({json:{configured:true}});
    }
    if(pathname==='/api/admin/train'){
      submissions.push({payload:request.postDataJSON(),authorization:request.headers().authorization});
      return route.fulfill({json:{id:'retry-1',status:'submitted'}});
    }
    return route.fulfill({status:404,json:{detail:'Fixture route unavailable'}});
  });
  await page.goto(base+'/observatory.html?api=/api');
  await page.waitForFunction(()=>document.getElementById('mode-badge').textContent.includes('Connected'));
  await page.locator('#setup-open').click();
  await page.locator('#owner-token').fill('fixture-owner-secret');
  // Explicit direct-provider fixture; the real default now uses x402/local.
  await page.locator('#setting-provider').selectOption('openai');
  await page.locator('#setting-browser').selectOption('browseruse');
  await page.locator('#setting-research-key').fill('fixture-teacher-secret');
  await page.locator('#setting-browser-key').fill('fixture-browser-secret');
  await page.locator('#setting-hf-token').fill('fixture-hub-secret');
  await page.locator('#settings-form [type="submit"]').click();
  await page.waitForFunction(()=>document.getElementById('setup-result').textContent.includes('Secret input fields cleared'));
  assert.equal(settingsRequests.length,1);
  assert.equal(settingsRequests[0].authorization,'Bearer fixture-owner-secret');
  assert.equal(settingsRequests[0].payload.openai_api_key,'fixture-teacher-secret');
  assert.equal(settingsRequests[0].payload.hf_token,'fixture-hub-secret');
  for(const field of ['setting-research-key','setting-browser-key','setting-hf-token'])assert.equal(await page.locator('#'+field).inputValue(),'');
  const storage=await page.evaluate(()=>JSON.stringify({...localStorage}));
  for(const secret of ['fixture-owner-secret','fixture-teacher-secret','fixture-browser-secret','fixture-hub-secret'])assert.ok(!storage.includes(secret));
  await page.locator('#setup-dialog [data-close-dialog]').click();
  await page.locator('[data-view="training"]').click();
  await page.locator('[data-retry-job="failed-1"]').click();
  assert.equal(submissions.length,0,'A retry must wait for explicit confirmation');
  await page.locator('#confirm-accept').click();
  await page.waitForTimeout(150);
  assert.equal(submissions.length,1);
  assert.deepEqual(submissions[0].payload,{snapshot_id:'sealed-1',retry:true,stage:'sft',parent_run_id:'cpt-1'});
  assert.equal(submissions[0].authorization,'Bearer fixture-owner-secret');
  await page.locator('[data-view="checkpoints"]').click();
  await page.locator('[data-checkpoint="selected-1"]').click();
  const downloadPromise=page.waitForEvent('download');
  await page.locator('[data-deployment-download="selected-1"]').click();
  const download=await downloadPromise, content=fs.readFileSync(await download.path(),'utf8');
  assert.match(content,/MODEL_ADAPTER_SUBFOLDER="runs\/job\/adapter"/);
  assert.ok(!content.includes('secret'));
  assert.equal(submissions.length,1,'Configuration download must not submit a job');
  assert.deepEqual(errors,[]);
  await browser.close();
  console.log(JSON.stringify({checks:'passed',defaultLive:true,ownerSettingsTransport:true,noSecretStorage:true,explicitSftRetry:true,secretFreeDeploymentDownload:true}));
})().catch(error=>{console.error(error);process.exit(1)});
