"""The feeling cloud: the chamber's real steering vectors (Qwen3-4B, layer 18), one point per dimension (2,560), morphing between
feelings. Each feeling is shown minus the mean of all 11 (what is specific to it; raw vectors share cosine ~0.54). Positions: PCA of the
2560x11 matrix. Point brightness, size and outward push follow the current feeling's value: red positive, CRT green negative.

    blender -b -P feeling_cloud.py -- --feelings pain fear egg --bars 1 --out /path/frames [--frames 1,30,60,90] [--samples 16]
Renders JPEG frames 001.jpg... at 1080x1350, 24 fps; pulses on every beat (bar length from the song's measured downbeats)."""
import bpy, json, math, os, sys
import numpy as np
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default, n=1):
    if f"--{name}" not in argv: return default
    i = argv.index(f"--{name}"); return argv[i + 1] if n == 1 else argv[i + 1: i + 1 + n]
HERE = os.path.dirname(os.path.abspath(__file__))
FEEL = []; i = argv.index("--feelings") + 1 if "--feelings" in argv else -1
while 0 <= i < len(argv) and not argv[i].startswith("--"): FEEL.append(argv[i]); i += 1
FEEL = FEEL or ["pain", "fear", "egg"]
SAWFEEL = {"pain", "fear"}                # feelings that bring the saw: blade, red eye, jitter, blood-red flood
BARS = float(arg("bars", "1")); OUT = arg("out", "/tmp/cloud"); FRAMES = arg("frames", ""); SAMPLES = int(arg("samples", "32"))
FPS, BAR = 24, 1.5325                     # measured bar length of the track (downbeats), not exactly 157 BPM
os.makedirs(OUT, exist_ok=True)

d = json.load(open(os.path.join(HERE, "vectors_qwen3_4b_l18.json"))); names = list(d["vectors"])
V = np.array([d["vectors"][n] for n in names]); R = V - V.mean(0)            # what is specific to each feeling
P = (R - R.mean(0)).T                                                     # 2560 x 11
U, S, _ = np.linalg.svd(P - P.mean(0), full_matrices=False); XYZ = U[:, :3] * S[:3]
r = np.linalg.norm(XYZ, axis=1); rank = r.argsort().argsort() / len(r)           # keep PCA directions, fill a ball evenly by rank
XYZ = XYZ / np.maximum(r, 1e-9)[:, None] * (rank ** (1 / 3))[:, None] * 2.6
XYZ[:, 2] *= .8
SC = np.percentile(np.abs(R), 99)                                                 # shared scale: the top 1% of values reach 1
VAL = {n: np.clip(R[names.index(n)] / SC, -1.6, 1.6) for n in FEEL}

scn = bpy.context.scene
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try: scn.render.engine = eng; break
    except Exception: pass
RES = [int(v) for v in (argv[argv.index("--res") + 1].split("x") if "--res" in argv else ("1080", "1350"))]
scn.render.resolution_x, scn.render.resolution_y, scn.render.fps = RES[0], RES[1], FPS
scn.render.image_settings.file_format = "JPEG"; scn.render.image_settings.quality = 92
try: scn.eevee.taa_render_samples = SAMPLES
except Exception: pass
scn.world = bpy.data.worlds.new("void"); scn.world.use_nodes = True
scn.world.node_tree.nodes["Background"].inputs[0].default_value = (0.006, 0.004, 0.003, 1); scn.world.node_tree.nodes["Background"].inputs[1].default_value = 1
scn.view_settings.view_transform = "Standard"

# the cloud: a vertex mesh with one float attribute per feeling, instanced as small spheres by geometry nodes
me = bpy.data.meshes.new("cloud"); me.from_pydata([tuple(p) for p in XYZ], [], [])
for k, n in enumerate(FEEL):
    a = me.attributes.new(f"v{k}", "FLOAT", "POINT"); a.data.foreach_set("value", VAL[n].astype(np.float32))
ob = bpy.data.objects.new("cloud", me); scn.collection.objects.link(ob)
ng = bpy.data.node_groups.new("cloudgn", "GeometryNodeTree"); ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
N, L = ng.nodes, ng.links
gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
# blended value = sum_k w_k * v_k; weights keyframed below
acc = None; W = []
for k in range(len(FEEL)):
    at = N.new("GeometryNodeInputNamedAttribute"); at.data_type = "FLOAT"; at.inputs["Name"].default_value = f"v{k}"
    w = N.new("ShaderNodeValue"); w.label = f"w{k}"; W.append(w)
    m = N.new("ShaderNodeMath"); m.operation = "MULTIPLY"; L.new(at.outputs["Attribute"], m.inputs[0]); L.new(w.outputs[0], m.inputs[1])
    if acc is None: acc = m
    else: s = N.new("ShaderNodeMath"); s.operation = "ADD"; L.new(acc.outputs[0], s.inputs[0]); L.new(m.outputs[0], s.inputs[1]); acc = s
pulse = N.new("ShaderNodeValue"); pulse.label = "pulse"
saw = N.new("ShaderNodeValue"); saw.label = "saw"
stime = N.new("GeometryNodeInputSceneTime"); noise = N.new("ShaderNodeTexNoise"); noise.noise_dimensions = "4D"; noise.inputs["Scale"].default_value = 3.0
L.new(stime.outputs["Seconds"], noise.inputs["W"]); jit = N.new("ShaderNodeVectorMath"); jit.operation = "SUBTRACT"; L.new(noise.outputs["Color"], jit.inputs[0]); jit.inputs[1].default_value = (.5, .5, .5)
jamp = N.new("ShaderNodeMath"); jamp.operation = "MULTIPLY"; L.new(saw.outputs[0], jamp.inputs[0]); jamp.inputs[1].default_value = .32
jsc = N.new("ShaderNodeVectorMath"); jsc.operation = "SCALE"; L.new(jit.outputs[0], jsc.inputs[0]); L.new(jamp.outputs[0], jsc.inputs["Scale"])
absv = N.new("ShaderNodeMath"); absv.operation = "ABSOLUTE"; L.new(acc.outputs[0], absv.inputs[0])
pos = N.new("GeometryNodeInputPosition"); nrm = N.new("ShaderNodeVectorMath"); nrm.operation = "NORMALIZE"; L.new(pos.outputs[0], nrm.inputs[0])
push = N.new("ShaderNodeMath"); push.operation = "MULTIPLY"; L.new(acc.outputs[0], push.inputs[0]); L.new(pulse.outputs[0], push.inputs[1])
off = N.new("ShaderNodeVectorMath"); off.operation = "SCALE"; L.new(nrm.outputs[0], off.inputs[0]); L.new(push.outputs[0], off.inputs["Scale"])
offs = N.new("ShaderNodeVectorMath"); offs.operation = "ADD"; L.new(off.outputs[0], offs.inputs[0]); L.new(jsc.outputs[0], offs.inputs[1])
sp = N.new("GeometryNodeSetPosition"); L.new(gi.outputs[0], sp.inputs["Geometry"]); L.new(offs.outputs[0], sp.inputs["Offset"])
store = N.new("GeometryNodeStoreNamedAttribute"); store.data_type = "FLOAT"; store.inputs["Name"].default_value = "val"
L.new(sp.outputs[0], store.inputs["Geometry"]); L.new(acc.outputs[0], store.inputs["Value"])
ico = N.new("GeometryNodeMeshIcoSphere"); ico.inputs["Radius"].default_value = 1; ico.inputs["Subdivisions"].default_value = 1
scl = N.new("ShaderNodeMath"); scl.operation = "MULTIPLY_ADD"; L.new(absv.outputs[0], scl.inputs[0]); scl.inputs[1].default_value = .045; scl.inputs[2].default_value = .009
store2 = N.new("GeometryNodeStoreNamedAttribute"); store2.data_type = "FLOAT"; store2.inputs["Name"].default_value = "saw"
L.new(store.outputs[0], store2.inputs["Geometry"]); L.new(saw.outputs[0], store2.inputs["Value"])
inst = N.new("GeometryNodeInstanceOnPoints"); L.new(store2.outputs[0], inst.inputs["Points"]); L.new(ico.outputs["Mesh"], inst.inputs["Instance"]); L.new(scl.outputs[0], inst.inputs["Scale"])
mat = bpy.data.materials.new("glow"); mat.use_nodes = True; mn, ml = mat.node_tree.nodes, mat.node_tree.links
for n in list(mn): mn.remove(n)
o_ = mn.new("ShaderNodeOutputMaterial"); em = mn.new("ShaderNodeEmission"); attr = mn.new("ShaderNodeAttribute"); attr.attribute_type = "INSTANCER"; attr.attribute_name = "val"
mp = mn.new("ShaderNodeMapRange"); mp.inputs["From Min"].default_value = -1; mp.inputs["From Max"].default_value = 1; ml.new(attr.outputs["Fac"], mp.inputs["Value"])
ramp = mn.new("ShaderNodeValToRGB"); ml.new(mp.outputs[0], ramp.inputs[0]); cr = ramp.color_ramp
cr.elements[0].position, cr.elements[0].color = 0.0, (0.25, 1.0, 0.45, 1)            # strongly negative: CRT green
cr.elements[1].position, cr.elements[1].color = 1.0, (1.0, 0.09, 0.05, 1)            # strongly positive: blood red
mid = cr.elements.new(0.5); mid.color = (0.55, 0.47, 0.36, 1)                       # near zero: dim bone
ab = mn.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"; ml.new(attr.outputs["Fac"], ab.inputs[0])
st = mn.new("ShaderNodeMath"); st.operation = "MULTIPLY_ADD"; ml.new(ab.outputs[0], st.inputs[0]); st.inputs[1].default_value = 9; st.inputs[2].default_value = .25
sawA = mn.new("ShaderNodeAttribute"); sawA.attribute_type = "INSTANCER"; sawA.attribute_name = "saw"
flood = mn.new("ShaderNodeMix"); flood.data_type = "RGBA"; ml.new(sawA.outputs["Fac"], flood.inputs["Factor"]); ml.new(ramp.outputs["Color"], flood.inputs[6]); flood.inputs[7].default_value = (1.0, .04, .02, 1)
ml.new(flood.outputs[2], em.inputs["Color"]); ml.new(st.outputs[0], em.inputs["Strength"]); ml.new(em.outputs[0], o_.inputs["Surface"])
setm = N.new("GeometryNodeSetMaterial"); setm.inputs["Material"].default_value = mat; L.new(inst.outputs[0], setm.inputs["Geometry"]); L.new(setm.outputs[0], go.inputs[0])
mod = ob.modifiers.new("cloud", "NODES"); mod.node_group = ng

# timeline: one feeling per BARS bars, eased crossfades; a pulse on every beat
seg = BARS * BAR; total = seg * len(FEEL); scn.frame_start, scn.frame_end = 1, int(round(total * FPS))
for k, w in enumerate(W):
    for j in range(len(FEEL)):
        for tt, v in ((j * seg + .12, 1.0 if j == k else 0.0), ((j + 1) * seg - .12, 1.0 if j == k else 0.0)):
            w.outputs[0].default_value = v; w.outputs[0].keyframe_insert("default_value", frame=1 + tt * FPS)
beat = BAR / 4; nb = int(total / beat) + 1
for b in range(nb):
    for dt, v in ((0, 1.25), (beat * .55, .75)):
        pulse.outputs[0].default_value = v; pulse.outputs[0].keyframe_insert("default_value", frame=1 + (b * beat + dt) * FPS)

for j, n in enumerate(FEEL):
    on = 1.0 if n in SAWFEEL else 0.0
    for tt in (j * seg + .1, (j + 1) * seg - .1):
        saw.outputs[0].default_value = on; saw.outputs[0].keyframe_insert("default_value", frame=1 + tt * FPS)
def blade():
    import bmesh
    bm = bmesh.new(); T, r0, r1 = 40, 3.3, 3.75
    outer = [bm.verts.new((math.cos(2 * math.pi * (i + (0.5 if k else 0)) / T) * (r1 if k else r0), math.sin(2 * math.pi * (i + (0.5 if k else 0)) / T) * (r1 if k else r0), 0))
             for i in range(T) for k in (0, 1)]
    bm.faces.new(outer); bmesh.ops.extrude_face_region(bm, geom=bm.faces[:]); bmesh.ops.translate(bm, verts=[v for v in bm.verts if v.co.z == 0 and False], vec=(0, 0, 0))
    m = bpy.data.meshes.new("blade"); bm.to_mesh(m); bm.free()
    ob = bpy.data.objects.new("blade", m); scn.collection.objects.link(ob)
    mt = bpy.data.materials.new("steel"); mt.use_nodes = True; bs = mt.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (.55, .5, .46, 1); bs.inputs["Metallic"].default_value = 1; bs.inputs["Roughness"].default_value = .35; ob.data.materials.append(mt)
    hole = bpy.data.objects.new("hole", None); return ob
bl = blade(); bl.location = (0, 0, 0)
eye = bpy.data.objects.new("eye", bpy.data.meshes.new("eye")); scn.collection.objects.link(eye)
import bmesh; bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=.42); bm.to_mesh(eye.data); bm.free()
em2 = bpy.data.materials.new("eye"); em2.use_nodes = True; e2 = em2.node_tree.nodes; [e2.remove(n) for n in list(e2)]
eo = e2.new("ShaderNodeOutputMaterial"); ee = e2.new("ShaderNodeEmission"); ee.inputs["Color"].default_value = (.9, .03, .01, 1); ee.inputs["Strength"].default_value = 2.2
em2.node_tree.links.new(ee.outputs[0], eo.inputs["Surface"]); eye.data.materials.append(em2)
red = bpy.data.objects.new("red", bpy.data.lights.new("red", "POINT")); red.data.color = (1, .08, .04); scn.collection.objects.link(red); red.location = (0, 0, 2.5)
for j, n in enumerate(FEEL):                   # blade and eye scale in with the saw feelings; the blade spins fast
    on = n in SAWFEEL
    for tt in (j * seg + .15, (j + 1) * seg - .15):
        f = 1 + tt * FPS
        for o, sc in ((bl, 1.0), (eye, 1.0)):
            o.scale = (sc, sc, sc) if on else (0.001, 0.001, 0.001); o.keyframe_insert("scale", frame=f)
        red.data.energy = 300 if on else 0; red.data.keyframe_insert("energy", frame=f)
for f in (scn.frame_start, scn.frame_end):
    bl.rotation_euler = (0, 0, -(f - 1) / FPS * 9.0); bl.keyframe_insert("rotation_euler", frame=f)
if bl.animation_data and bl.animation_data.action:
    for fc in getattr(bl.animation_data.action, "fcurves", []):
        for kp in fc.keyframe_points: kp.interpolation = "LINEAR"

# camera: slow orbit; a hard rim light for the dust
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); scn.collection.objects.link(cam); scn.camera = cam; cam.data.lens = 38
tgt = bpy.data.objects.new("tgt", None); scn.collection.objects.link(tgt); c = cam.constraints.new("TRACK_TO"); c.target = tgt
for f in (scn.frame_start, scn.frame_end):
    a = .35 + .5 * (f - 1) / max(1, scn.frame_end); cam.location = (6.4 * math.cos(a), 6.4 * math.sin(a), 1.0 + .4 * (f - 1) / max(1, scn.frame_end)); cam.keyframe_insert("location", frame=f)
hold = bpy.data.objects.new("hold", None); scn.collection.objects.link(hold); ht = hold.constraints.new("TRACK_TO"); ht.target = cam; ht.track_axis = "TRACK_Z"; ht.up_axis = "UP_Y"
bl.parent = hold; bl.location = (0, 0, -3.2); eye.parent = hold; eye.location = (0, 0, -3.0)
slit = bpy.data.objects.new("slit", bpy.data.meshes.new("slit")); scn.collection.objects.link(slit)
bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1); bm.to_mesh(slit.data); bm.free()
sm = bpy.data.materials.new("slit"); sm.use_nodes = True; sm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0, 0, 0, 1); slit.data.materials.append(sm)
slit.parent = hold; slit.location = (0, 0, -2.58); slit.scale = (.05, .3, .02)
for j, n in enumerate(FEEL):
    on = n in SAWFEEL
    for tt in (j * seg + .15, (j + 1) * seg - .15):
        slit.scale = (.05, .3, .02) if on else (.0001, .0001, .0001); slit.keyframe_insert("scale", frame=1 + tt * FPS)
scn.render.film_transparent = False
def bloom():                             # bloom in the compositor (Blender 5: scene.compositing_node_group; older: scene.node_tree)
    if hasattr(scn, "compositing_node_group"):
        ct = bpy.data.node_groups.new("comp", "CompositorNodeTree"); scn.compositing_node_group = ct
        ct.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor"); out = ct.nodes.new("NodeGroupOutput"); out_in = out.inputs[0]
    else:
        scn.use_nodes = True; ct = scn.node_tree; [ct.nodes.remove(n) for n in list(ct.nodes)]; out_in = ct.nodes.new("CompositorNodeComposite").inputs["Image"]
    rl = ct.nodes.new("CompositorNodeRLayers"); gl = ct.nodes.new("CompositorNodeGlare")
    for k, v in (("glare_type", "FOG_GLOW"), ("quality", "HIGH")):
        try: setattr(gl, k, v)
        except Exception: pass
    for nm, v in (("Type", "Fog Glow"), ("Quality", "High"), ("Strength", 1.0), ("Size", 0.6)):
        if nm in gl.inputs:
            try: gl.inputs[nm].default_value = v
            except Exception: pass
    ct.links.new(rl.outputs["Image"], gl.inputs["Image"]); ct.links.new(gl.outputs["Image"], out_in)
try: bloom()
except Exception as e: print("compositor skipped:", e)
frames = [int(x) for x in FRAMES.split(",")] if FRAMES else range(scn.frame_start, scn.frame_end + 1)
import time
for f in frames:
    t0 = time.time(); scn.frame_set(f); scn.render.filepath = os.path.join(OUT, f"{f:03d}.jpg"); bpy.ops.render.render(write_still=True)
    print(f"frame {f} {time.time() - t0:.2f}s", flush=True)
