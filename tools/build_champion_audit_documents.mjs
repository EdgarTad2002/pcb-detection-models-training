/** Rebuild the workbook and ten-slide presentation only from a completed audit. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Workbook, SpreadsheetFile, Presentation, PresentationFile} from '@oai/artifact-tool';

// The finalizer launches its own import-verification process and needs the
// bundled dependency location in addition to this builder's node_modules link.
process.env.RUNTIME_NODE_MODULES ??= '/home/edgar/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';

const [input, destination] = process.argv.slice(2);
if (!input || !destination) throw new Error('Usage: build_champion_audit_documents.mjs SUMMARY_JSON OUTPUT_DIR');
const data = JSON.parse(await fs.readFile(input, 'utf8'));
const {manifest, results, hardware} = data;
const cfg = manifest.config;
const names = manifest.names;
const expected = [...cfg.models, ...cfg.ensembles];
if (data.status !== 'completed' || results.length !== 6 || new Set(results.map(r=>r.id)).size !== 6)
  throw new Error('Require all six completed configurations');
for (const c of expected) {
  const r = results.find(x=>x.id===c.id);
  if (!r || r.manifest_id !== manifest.id) throw new Error('Unmatched result provenance');
  const members = c.members ?? [c.id];
  if (JSON.stringify(r.checkpoint_sha256) !== JSON.stringify(members.map(id=>cfg.models.find(m=>m.id===id).sha256)))
    throw new Error('Result belongs to another checkpoint');
  if (!(r.accuracy.mAP50>=0 && r.accuracy.mAP50<=1 && r.speed.wall_prediction.mean_ms>0))
    throw new Error('Missing or invalid accuracy/timing');
}
await fs.mkdir(destination, {recursive:true});
const workspaceDir = path.resolve(destination, '../..');
const privateDir = path.join(workspaceDir, '.build', path.basename(destination));
await fs.mkdir(privateDir, {recursive:true});
const pct = v => `${(100*v).toFixed(2)}%`;
const sorted = [...results].sort((a,b)=>b.accuracy.mAP50-a.accuracy.mAP50);
const sourceHash = crypto.createHash('sha256').update(await fs.readFile(input)).digest('hex');

const workbook = Workbook.create();
function sheet(name, title, context, headers, rows, widths) {
  const s=workbook.worksheets.add(name);
  s.showGridLines=false;
  s.getRange('A2').values=[[title]];
  s.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:'#0B1F3A'};
  s.getRange('A3').values=[[context]];
  s.getRangeByIndexes(4,0,rows.length+1,headers.length).values=[headers,...rows];
  const table=s.getRangeByIndexes(4,0,rows.length+1,headers.length);
  table.format.font={name:'Arial',size:11,color:'#152D47'};
  table.format.rowHeight=35;
  table.format.verticalAlignment='center';
  table.format.wrapText=true;
  const heading=s.getRangeByIndexes(4,0,1,headers.length);
  heading.format.fill='#0B1F3A';
  heading.format.font={name:'Arial',size:11,bold:true,color:'#FFFFFF'};
  heading.format.rowHeight=45;
  widths.forEach((w,i)=>s.getRangeByIndexes(0,i,rows.length+5,1).format.columnWidthPx=w);
  rows.forEach((_,i)=>{if(i%2===0)s.getRangeByIndexes(i+5,0,1,headers.length).format.fill='#EDF3F8';});
  s.freezePanes.freezeRows(5);
  return s;
}
const overall=sheet('Master Benchmark & Ensembles','PCB detection: verified checkpoint benchmark',
  `${manifest.records.length} cleaned test boards. FP32, 1280 px, batch 1. Recall: confidence >= 0.25, matching IoU >= 0.50.`,
  ['Configuration','Members','Fusion','mAP50 (%)','mAP50:95 (%)','Cap AP50 (%)','Cap recall (%)','Conn AP50 (%)','Elect AP50 (%)','IC AP50 (%)','Mean FP32 ms','P95 FP32 ms','FP32 FPS'],
  sorted.map(r=>[r.name,r.members.join(' + '),r.members.length>1?'WBF 0.50':'Single',100*r.accuracy.mAP50,100*r.accuracy.mAP50_95,
    ...[names[0]].map(n=>100*r.accuracy.per_class[n].ap50),100*r.accuracy.per_class[names[0]].recall,
    ...names.slice(1).map(n=>100*r.accuracy.per_class[n].ap50),r.speed.wall_prediction.mean_ms,r.speed.wall_prediction.p95_ms,null]),
  [290,255,110,110,120,110,110,115,115,110,115,115,105]);
overall.getRange('D6:M11').setNumberFormat('0.00');
sorted.forEach((r,i)=>overall.getRange(`M${i+6}`).formulas=[[`=1000/K${i+6}`]]);
overall.getRange('A14').values=[['Selected historical checkpoints; results do not establish a controlled training comparison.']];
overall.getRange('A15').values=[['Source: '+path.resolve(input)]];
overall.getRange('A16').values=[['Source SHA256: '+sourceHash]];

const detailRows=cfg.models.map(m=>{
  const r=results.find(r=>r.id===m.id);
  return [m.name,m.path,m.sha256,r.parameters[0]];
});
const details=sheet('Model Weights & Details','Exact checkpoint files','Each model passed inference; hashes identify the evaluated binary files.',
  ['Model','Checkpoint path','SHA256','Inference parameters'],detailRows,[290,740,580,165]);
details.getRange('D6:D9').setNumberFormat('#,##0');
const protocolRows=[['Dataset',cfg.data],['Test fingerprint',manifest.test_fingerprint],['Manifest ID',manifest.id],
 ['Accuracy',manifest.metric],['Inference',JSON.stringify(cfg.settings)],['Speed scope',manifest.speed_scope],
 ['GPU',hardware.gpu??'CPU'],['Frameworks',JSON.stringify(manifest.versions)],
 ['Parameter counts','Recorded after predictor initialization and deployment fusion; these differ from unfused training-checkpoint counts.'],['Disclosure',manifest.disclosure]];
details.getRangeByIndexes(12,0,protocolRows.length,2).values=protocolRows;
details.getRangeByIndexes(12,0,protocolRows.length,2).format.wrapText=true;
details.getRangeByIndexes(12,0,protocolRows.length,2).format.rowHeight=42;
details.getRange('A17:B17').format.rowHeight=84;
const perClass=sheet('Per-Class Performance','Per-class accuracy and operating counts',
  'AP integrates confidence scores. TP, FP, FN, precision and recall use confidence >= 0.25 and matching IoU >= 0.50.',
  ['Configuration','Class','AP50 (%)','AP50:95 (%)','TP','FP','FN','Ground truth','Recall (%)','Precision (%)'],
  sorted.flatMap(r=>names.map(n=>{const a=r.accuracy.per_class[n];return [r.name,n,100*a.ap50,100*a.ap50_95,a.tp,a.fp,a.fn,a.targets,a.recall===null?null:100*a.recall,100*a.precision];})),
  [290,210,115,120,85,85,85,115,115,115]);
perClass.getRange('C6:D29').setNumberFormat('0.00');
perClass.getRange('I6:J29').setNumberFormat('0.00');
await workbook.recalculate();
console.log((await workbook.inspect({kind:'region',sheetId:overall.name,range:'D5:M11',maxChars:2500,tableMaxRows:7,tableMaxCols:10})).ndjson);
const errors = await workbook.inspect({kind:'match', searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100}});
await fs.writeFile(path.join(privateDir,'workbook-formula-scan.ndjson'),errors.ndjson);
await (await SpreadsheetFile.exportXlsx(workbook)).save(path.join(destination,'PCB_Detection_Master_Benchmark_Verified.xlsx'));
const workbookInspection = path.join(destination,'PCB_Detection_Master_Benchmark_Verified.xlsx.inspect.ndjson');
try { await fs.rename(workbookInspection,path.join(privateDir,'workbook-inspection.ndjson')); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
await fs.writeFile(path.join(privateDir,'workbook.png'),new Uint8Array(await (await workbook.render({sheetName:overall.name,range:'A2:M11',scale:1,format:'png'})).arrayBuffer()));
await fs.writeFile(path.join(privateDir,'workbook-details.png'),new Uint8Array(await (await workbook.render({sheetName:details.name,range:'A2:D22',scale:1,format:'png'})).arrayBuffer()));
await fs.writeFile(path.join(privateDir,'workbook-per-class.png'),new Uint8Array(await (await workbook.render({sheetName:perClass.name,range:'A2:J29',scale:1,format:'png'})).arrayBuffer()));

const presentation=Presentation.create({slideSize:{width:1280,height:720}});
presentation.theme.defaultFont='Arial';
const short={yolo12:'YOLOv12s',crop:'YOLO26s crop',agwd_sdice:'AGWD + Squared-Dice',physot:'PhysOT-YOLO26s',ensemble2:'Two-model WBF',ensemble4:'Four-model WBF'};
function text(slide,value,left,top,width,height,size=24,color='#152D47',bold=false){
  const s=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
  s.text=value;s.text.style={typeface:'Arial',fontSize:size,color,bold};return s;
}
function page(title,subtitle=''){
  const s=presentation.slides.add();s.background.fill='#FFFFFF';
  text(s,title,56,40,1168,85,40,'#0B1F3A',true);
  if(subtitle)text(s,subtitle,56,130,1168,70,22);
  text(s,`PCB component detection    ${presentation.slides.items.length} / 10`,56,672,1168,24,16,'#52677B');
  s.speakerNotes.text=`Source: ${path.resolve(input)}\nSHA256: ${sourceHash}\nManifest: ${manifest.id}\n${manifest.disclosure}`;
  return s;
}
function table(slide,headers,rows,widths,top=218,height=390){
  const t=slide.tables.add({rows:rows.length+1,columns:headers.length,left:56,top,width:1168,height,
    columnWidths:widths,values:[headers,...rows]});
  t.borders.assign({style:'solid',fill:'#D5DEE7',width:.6});
  for(let row=0;row<=rows.length;row++)for(let col=0;col<headers.length;col++){
    const cell=t.getCell(row,col);cell.fill=row===0?'#0B1F3A':row%2?'#EDF3F8':'#FFFFFF';
    cell.text.style={typeface:'Arial',fontSize:20,color:row===0?'#FFFFFF':'#152D47',bold:row===0};
  }
  return t;
}
let s=page('PCB detection benchmark','Four saved single models and two fixed WBF configurations');
text(s,`${manifest.records.length} cleaned test boards`,56,255,1168,70,42,'#0B1F3A',true);
text(s,'Exact checkpoint files. One metric implementation. Measured inference speed.',56,355,1100,90,30);
text(s,'Historical native-resolution experiment at 1280 pixels',56,495,1100,60,25);

s=page('Evaluation protocol','The same input processing and AP calculation apply to all six configurations');
table(s,['Setting','Value'],[['Test set',`${manifest.records.length} full-board images, cleaned labels`],
 ['Inference','1280 px square letterbox, batch 1, FP32'],['Member confidence floor','0.001, maximum 1000 detections per board'],
 ['NMS and WBF','NMS IoU 0.45 where applicable; WBF IoU 0.50'],['AP','Ultralytics matching, IoU 0.50 to 0.95'],
 ['Operating recall','Confidence 0.25, matching IoU 0.50']],[330,838],210,420);

s=page('Evaluated checkpoints','Inference parameter counts follow predictor initialization and deployment fusion');
table(s,['Model','Inference params','Checkpoint file'],cfg.models.map(m=>{const r=results.find(x=>x.id===m.id);return [short[m.id],r.parameters[0].toLocaleString('en-US'),path.basename(m.path)];}),[340,200,628],230,320);
text(s,'The crop specialist uses a single full-board inference pass in this comparison.',56,580,1150,60,23);

s=page('Overall detection accuracy','Percent AP, computed from the same saved test predictions');
table(s,['Configuration','mAP50','mAP50:95'],sorted.map(r=>[short[r.id],pct(r.accuracy.mAP50),pct(r.accuracy.mAP50_95)]),[568,300,300]);

s=page('Per-class detection accuracy','AP50 reflects both precision and recall over confidence thresholds');
table(s,['Configuration','Capacitor','Connector','Electrolytic','IC'],sorted.map(r=>[short[r.id],...names.map(n=>pct(r.accuracy.per_class[n].ap50))]),[408,190,190,190,190]);

s=page('Capacitor recall at confidence 0.25','Class-aware matching at IoU 0.50, with a fixed operating confidence');
table(s,['Configuration','Recall','TP','FP','Missed'],sorted.map(r=>{const a=r.accuracy.per_class[names[0]];return [short[r.id],pct(a.recall),String(a.tp),String(a.fp),String(a.fn)];}),[408,190,190,190,190]);

s=page('Measured inference speed',`${hardware.gpu??'CPU'}: FP32, batch 1, ${cfg.settings.warmup} warmups and ${cfg.settings.iterations} timed iterations`);
table(s,['Configuration','Mean ms','P95 ms','FPS'],sorted.map(r=>[short[r.id],r.speed.wall_prediction.mean_ms.toFixed(2),r.speed.wall_prediction.p95_ms.toFixed(2),r.speed.fps_from_mean_latency.toFixed(2)]),[508,220,220,220]);
text(s,'Includes preprocessing, detector postprocessing and WBF. Excludes disk decoding.',56,619,1168,35,19);

for(const id of ['ensemble2','ensemble4']){
  const r=results.find(r=>r.id===id);
  s=page(id==='ensemble2'?'Two-model ensemble':'Four-model ensemble','Fixed weighted box fusion at IoU 0.50');
  table(s,['Member','Fusion weight'],r.members.map((m,i)=>[short[m],r.weights[i].toFixed(1)]),[838,330],220,id==='ensemble2'?180:280);
  text(s,`mAP50 ${pct(r.accuracy.mAP50)}     mAP50:95 ${pct(r.accuracy.mAP50_95)}`,56,535,1168,60,30,'#0B1F3A',true);
  text(s,'Weights and thresholds were fixed before this verification run.',56,612,1168,36,22);
}
s=page('Scope of these results','Evidence for the selected saved checkpoints on the cleaned native-resolution test set');
text(s,'The models used different training recipes. This benchmark compares their saved outputs and deployment speed.',56,235,1140,105,28);
text(s,'Historical test results informed development. This test set is not a newly untouched holdout.',56,365,1140,100,28);
text(s,'Comparison with Information 17, 322 requires the separate paper-dataset protocol and its own results.',56,500,1140,110,28);

for(const [i,slide] of presentation.slides.items.entries()){
  const png=await presentation.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(privateDir,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
const candidatePath=path.join(privateDir,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const skill='/home/edgar/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11618/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
await finalizePresentation({workspaceDir,candidatePath,finalPath:path.join(path.resolve(destination),'Professor_Meeting_Presentation_Verified.pptx'),
 pythonExecutable:'/home/edgar/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',
   ...[2,3,4,5,6,7,8,9].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:10,requiredNativeTableOwnerSlides:[2,3,4,5,6,7,8,9],
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
 receiptPath:path.join(privateDir,'presentation-validation.json')});
console.log('Verified documents rebuilt from '+path.resolve(input));
