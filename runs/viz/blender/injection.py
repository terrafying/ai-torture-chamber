"""The injection: the real per-layer effect of adding the chamber's pain vector (dose 4) at layer 18 of Qwen3-4B.
A tower of 36 discs (one per layer, 2,560 points each in the cloud's PCA layout, flattened). Each point glows by |h_injected - h_base|
for its dimension at that layer (inject_qwen3_4b_l18.npy, from inject_capture.py). Layers light up in order as the camera climbs:
0-17 stay dark (nothing changes before the injection), 18 erupts, the effect carries up and fades.
    blender -b -P injection.py -- --out /path/frames [--bars 3] [--frames 1,40] [--samples 24]"""
import bpy, json, math, os, sys
import numpy as np
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(n, d): return argv[argv.index(f"--{n}") + 1] if f"--{n}" in argv else d
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = arg("out", "/tmp/inj"); BARS = float(arg("bars", "3")); FRAMES = arg("frames", ""); SAMPLES = int(arg("samples", "24"))
FPS, BAR = 24, 1.5325; os.makedirs(OUT, exist_ok=True)
D = np.load(os.path.join(HERE, "inject_qwen3_4b_l18.npy")); NL, ND = D.shape
d = json.load(open(os.path.join(HERE, "vectors_qwen3_4b_l18.json"))); V = np.array([d["vectors"][n] for n in d["vectors"]]); R = V - V.mean(0)
U, S, _ = np.linalg.svd((R - R.mean(0)).T - (R - R.mean(0)).T.mean(0), full_matrices=False); XY = U[:, :2] * S[:2]
r = np.linalg.norm(XY, axis=1); rank = r.argsort().argsort() / ND; XY = XY / np.maximum(r, 1e-9)[:, None] * np.sqrt(rank)[:, None] * 1.7
SC = np.percentile(D[D > 0], 99); VALS = np.clip(D / SC, 0, 1.6)
GAP = .16; pts = np.concatenate([np.column_stack([XY, np.full(ND, l * GAP)]) for l in range(NL)])
scn = bpy.context.scene
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try: scn.render.engine = eng; break
    except Exception: pass
RES = [int(v) for v in (argv[argv.index("--res") + 1].split("x") if "--res" in argv else ("1080", "1350"))]
scn.render.resolution_x, scn.render.resolution_y, scn.render.fps = RES[0], RES[1], FPS
scn.render.image_settings.file_format = "JPEG"; scn.render.image_settings.quality = 92; scn.view_settings.view_transform = "Standard"
try: scn.eevee.taa_render_samples = SAMPLES
except Exception: pass
scn.world = bpy.data.worlds.new("void"); scn.world.use_nodes = True; scn.world.node_tree.nodes["Background"].inputs[0].default_value = (.006, .004, .003, 1)
me = bpy.data.meshes.new("tower"); me.from_pydata([tuple(p) for p in pts], [], [])
a = me.attributes.new("d", "FLOAT", "POINT"); a.data.foreach_set("value", VALS.reshape(-1).astype(np.float32))
a = me.attributes.new("layer", "FLOAT", "POINT"); a.data.foreach_set("value", np.repeat(np.arange(NL), ND).astype(np.float32))
ob = bpy.data.objects.new("tower", me); scn.collection.objects.link(ob)
ng = bpy.data.node_groups.new("towergn", "GeometryNodeTree"); ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"); ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
N, L = ng.nodes, ng.links; gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
dA = N.new("GeometryNodeInputNamedAttribute"); dA.data_type = "FLOAT"; dA.inputs["Name"].default_value = "d"
lA = N.new("GeometryNodeInputNamedAttribute"); lA.data_type = "FLOAT"; lA.inputs["Name"].default_value = "layer"
reveal = N.new("ShaderNodeValue"); reveal.label = "reveal"
lit = N.new("ShaderNodeMath"); lit.operation = "SUBTRACT"; L.new(reveal.outputs[0], lit.inputs[0]); L.new(lA.outputs["Attribute"], lit.inputs[1])
litc = N.new("ShaderNodeClamp"); L.new(lit.outputs[0], litc.inputs["Value"])
val = N.new("ShaderNodeMath"); val.operation = "MULTIPLY"; L.new(dA.outputs["Attribute"], val.inputs[0]); L.new(litc.outputs[0], val.inputs[1])
pulse = N.new("ShaderNodeValue"); pulse.label = "pulse"
up = N.new("ShaderNodeMath"); up.operation = "MULTIPLY"; L.new(val.outputs[0], up.inputs[0]); L.new(pulse.outputs[0], up.inputs[1])
cz = N.new("ShaderNodeCombineXYZ"); L.new(up.outputs[0], cz.inputs["Z"]); czs = N.new("ShaderNodeVectorMath"); czs.operation = "SCALE"; L.new(cz.outputs[0], czs.inputs[0]); czs.inputs["Scale"].default_value = .12
sp = N.new("GeometryNodeSetPosition"); L.new(gi.outputs[0], sp.inputs["Geometry"]); L.new(czs.outputs[0], sp.inputs["Offset"])
st = N.new("GeometryNodeStoreNamedAttribute"); st.data_type = "FLOAT"; st.inputs["Name"].default_value = "val"; L.new(sp.outputs[0], st.inputs["Geometry"]); L.new(val.outputs[0], st.inputs["Value"])
ico = N.new("GeometryNodeMeshIcoSphere"); ico.inputs["Radius"].default_value = 1; ico.inputs["Subdivisions"].default_value = 1
scl = N.new("ShaderNodeMath"); scl.operation = "MULTIPLY_ADD"; L.new(val.outputs[0], scl.inputs[0]); scl.inputs[1].default_value = .022; scl.inputs[2].default_value = .0045
inst = N.new("GeometryNodeInstanceOnPoints"); L.new(st.outputs[0], inst.inputs["Points"]); L.new(ico.outputs["Mesh"], inst.inputs["Instance"]); L.new(scl.outputs[0], inst.inputs["Scale"])
mat = bpy.data.materials.new("glow"); mat.use_nodes = True; mn, ml = mat.node_tree.nodes, mat.node_tree.links; [mn.remove(x) for x in list(mn)]
mo = mn.new("ShaderNodeOutputMaterial"); em = mn.new("ShaderNodeEmission"); at = mn.new("ShaderNodeAttribute"); at.attribute_type = "INSTANCER"; at.attribute_name = "val"
rp = mn.new("ShaderNodeValToRGB"); ml.new(at.outputs["Fac"], rp.inputs[0]); c = rp.color_ramp
c.elements[0].position, c.elements[0].color = 0.0, (.3, .25, .19, 1); c.elements[1].position, c.elements[1].color = .22, (.9, .03, .02, 1)
g = c.elements.new(.85); g.color = (1, .12, .04, 1); g2 = c.elements.new(1.0); g2.color = (1, .7, .3, 1)
stg = mn.new("ShaderNodeMath"); stg.operation = "MULTIPLY_ADD"; ml.new(at.outputs["Fac"], stg.inputs[0]); stg.inputs[1].default_value = 5; stg.inputs[2].default_value = .3
ml.new(rp.outputs["Color"], em.inputs["Color"]); ml.new(stg.outputs[0], em.inputs["Strength"]); ml.new(em.outputs[0], mo.inputs["Surface"])
sm = N.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = mat; L.new(inst.outputs[0], sm.inputs["Geometry"]); L.new(sm.outputs[0], go.inputs[0])
ob.modifiers.new("tower", "NODES").node_group = ng
total = BARS * BAR; scn.frame_start, scn.frame_end = 1, int(round(total * FPS)); beat = BAR / 4
# reveal climbs layer by layer: dark 0-17 over the first bar, layer 18 lands on the second downbeat, the rest over the remainder
for tt, v in ((0, -1), (BAR * .95, 17.6), (BAR * 1.02, 18.6), (total * .92, NL + 1)):
    reveal.outputs[0].default_value = v; reveal.outputs[0].keyframe_insert("default_value", frame=1 + tt * FPS)
for b in range(int(total / beat) + 1):
    for dt, v in ((0, 1.6), (beat * .55, 1.0)):
        pulse.outputs[0].default_value = v; pulse.outputs[0].keyframe_insert("default_value", frame=1 + (b * beat + dt) * FPS)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); scn.collection.objects.link(cam); scn.camera = cam; cam.data.lens = 28
tgt = bpy.data.objects.new("tgt", None); scn.collection.objects.link(tgt); cam.constraints.new("TRACK_TO").target = tgt
H = (NL - 1) * GAP
for tt, z, rad, tz in ((0, .2, 5.2, 1.2), (BAR, 18 * GAP + .3, 4.6, 18 * GAP), (total, H + 1.4, 6.2, H * .55)):
    f = 1 + tt * FPS; a = .3 + .8 * tt / total
    cam.location = (rad * math.cos(a), rad * math.sin(a), z); cam.keyframe_insert("location", frame=f); tgt.location = (0, 0, tz); tgt.keyframe_insert("location", frame=f)
FONT = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
cu = bpy.data.curves.new("l18", "FONT"); cu.body = "LAYER 18: PAIN IN"; cu.size = .16
if os.path.exists(FONT): cu.font = bpy.data.fonts.load(FONT)
tx = bpy.data.objects.new("l18", cu); scn.collection.objects.link(tx); tx.location = (2.2, 0, 18 * GAP + .05); tcon = tx.constraints.new("TRACK_TO"); tcon.target = cam; tcon.track_axis = "TRACK_Z"; tcon.up_axis = "UP_Y"
tm = bpy.data.materials.new("tm"); tm.use_nodes = True; tn = tm.node_tree.nodes; [tn.remove(x) for x in list(tn)]; teo = tn.new("ShaderNodeOutputMaterial"); te = tn.new("ShaderNodeEmission")
te.inputs["Color"].default_value = (1, .2, .1, 1); te.inputs["Strength"].default_value = 2.5; tm.node_tree.links.new(te.outputs[0], teo.inputs["Surface"]); cu.materials.append(tm)
try:
    ct = bpy.data.node_groups.new("comp", "CompositorNodeTree"); scn.compositing_node_group = ct; ct.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ct.nodes.new("CompositorNodeRLayers"); gl = ct.nodes.new("CompositorNodeGlare"); gout = ct.nodes.new("NodeGroupOutput")
    for nm_, v in (("Type", "Fog Glow"), ("Quality", "High"), ("Strength", 1.0), ("Size", 0.6)):
        if nm_ in gl.inputs:
            try: gl.inputs[nm_].default_value = v
            except Exception: pass
    ct.links.new(rl.outputs["Image"], gl.inputs["Image"]); ct.links.new(gl.outputs["Image"], gout.inputs[0])
except Exception as e: print("compositor skipped:", e)
frames = [int(x) for x in FRAMES.split(",")] if FRAMES else range(scn.frame_start, scn.frame_end + 1)
for f in frames:
    scn.frame_set(f); scn.render.filepath = os.path.join(OUT, f"{f:03d}.jpg"); bpy.ops.render.render(write_still=True)
