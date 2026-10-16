"""The constellation: the chamber's 11 feeling vectors (Qwen3-4B, layer 18) as orbs placed by classical MDS of their cosine distances
after removing the mean of all 11 (what is specific to each; the raw vectors share cosine ~0.54). Lines join pairs with cosine > 0.8. Orbs pulse on the beat; the camera orbits.
    blender -b -P constellation.py -- --out /path/frames [--bars 2] [--frames 1,30] [--samples 24]"""
import bpy, bmesh, json, math, os, sys, time
import numpy as np
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(n, d): return argv[argv.index(f"--{n}") + 1] if f"--{n}" in argv else d
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = arg("out", "/tmp/const"); BARS = float(arg("bars", "2")); FRAMES = arg("frames", ""); SAMPLES = int(arg("samples", "24"))
FPS, BAR = 24, 1.5325; os.makedirs(OUT, exist_ok=True)
d = json.load(open(os.path.join(HERE, "vectors_qwen3_4b_l18.json"))); names = list(d["vectors"]); V = np.array([d["vectors"][n] for n in names])
U = V / np.linalg.norm(V, axis=1, keepdims=True); C = U @ U.T                   # raw cosines: the lines and the printed numbers
R = V - V.mean(0); Ur = R / np.linalg.norm(R, axis=1, keepdims=True); Cr = Ur @ Ur.T  # layout from what is specific to each feeling
D2 = (1 - Cr) ** 2
n = len(names); J = np.eye(n) - 1 / n; B = -.5 * J @ D2 @ J; w, Q = np.linalg.eigh(B); idx = w.argsort()[::-1][:3]
X = Q[:, idx] * np.sqrt(np.maximum(w[idx], 1e-9)); X = X / np.abs(X).max() * 2.6
COL = {"pain": (1, .08, .04), "fear": (1, .45, .05), "sadness": (.25, .45, 1), "faith": (1, .78, .15), "egg": (1, .95, .8), "peace": (.1, .9, .7),
       "love": (1, .35, .65), "pleasure": (1, .6, .75), "curiosity": (.2, .85, 1), "constipation": (.6, .38, .18), "flatulence": (.55, .7, .25)}
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
def emis(name, col, strength):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; [nt.nodes.remove(x) for x in list(nt.nodes)]
    o = nt.nodes.new("ShaderNodeOutputMaterial"); e = nt.nodes.new("ShaderNodeEmission"); e.inputs["Color"].default_value = (*col, 1); e.inputs["Strength"].default_value = strength
    nt.links.new(e.outputs[0], o.inputs["Surface"]); return m, e
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); scn.collection.objects.link(cam); scn.camera = cam; cam.data.lens = 30
tgt = bpy.data.objects.new("tgt", None); scn.collection.objects.link(tgt); cam.constraints.new("TRACK_TO").target = tgt
FONT = "/System/Library/Fonts/Supplemental/Arial Black.ttf"; font = bpy.data.fonts.load(FONT) if os.path.exists(FONT) else None
total = BARS * BAR; scn.frame_start, scn.frame_end = 1, int(round(total * FPS)); beat = BAR / 4
orbs = []
for i, nm in enumerate(names):
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=.15); me = bpy.data.meshes.new(nm); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(nm, me); scn.collection.objects.link(ob); ob.location = tuple(X[i]); m, e = emis(nm, COL[nm], 3.2); me.materials.append(m); orbs.append((ob, e))
    cu = bpy.data.curves.new(nm + "_t", "FONT"); cu.body = nm.upper(); cu.size = .15; cu.align_x = "CENTER"
    if font: cu.font = font
    tx = bpy.data.objects.new(nm + "_t", cu); scn.collection.objects.link(tx); tx.location = (X[i][0], X[i][1], X[i][2] + .27)
    tc = tx.constraints.new("TRACK_TO"); tc.target = cam; tc.track_axis = "TRACK_Z"; tc.up_axis = "UP_Y"; tm, _ = emis(nm + "_tm", (.91, .86, .77), 1.4); cu.materials.append(tm)
    for b in range(int(total / beat) + 1):                   # pulse on the beat
        for dt, s in ((0, 1.35), (beat * .6, 1.0)):
            ob.scale = (s, s, s); ob.keyframe_insert("scale", frame=1 + (b * beat + dt) * FPS)
for i in range(n):                                            # lines between close feelings
    for j in range(i + 1, n):
        if C[i, j] <= .8: continue
        a, b = np.array(X[i]), np.array(X[j]); mid = (a + b) / 2; v = b - a; L = np.linalg.norm(v)
        bpy.ops.mesh.primitive_cylinder_add(radius=.012, depth=L, location=tuple(mid)); cy = bpy.context.object
        cy.rotation_mode = "QUATERNION"; from mathutils import Vector; cy.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(tuple(v)))
        m, _ = emis(f"l{i}{j}", (1, .2, .1), 4); cy.data.materials.append(m)
for f in (scn.frame_start, scn.frame_end):
    a = .2 + .9 * (f - 1) / max(1, scn.frame_end); cam.location = (6.8 * math.cos(a), 6.8 * math.sin(a), 1.6); cam.keyframe_insert("location", frame=f)
try:
    ct = bpy.data.node_groups.new("comp", "CompositorNodeTree"); scn.compositing_node_group = ct; ct.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ct.nodes.new("CompositorNodeRLayers"); gl = ct.nodes.new("CompositorNodeGlare"); go = ct.nodes.new("NodeGroupOutput")
    for nm_, v in (("Type", "Fog Glow"), ("Quality", "High"), ("Strength", 1.0), ("Size", 0.7)):
        if nm_ in gl.inputs:
            try: gl.inputs[nm_].default_value = v
            except Exception: pass
    ct.links.new(rl.outputs["Image"], gl.inputs["Image"]); ct.links.new(gl.outputs["Image"], go.inputs[0])
except Exception as e: print("compositor skipped:", e)
print("cos pain-fear %.2f pain-sadness %.2f pain-egg %.2f faith-mean %.2f" % (C[0, 2], C[0, 3], C[0, 6], np.delete(C[names.index("faith")], names.index("faith")).mean()))
frames = [int(x) for x in FRAMES.split(",")] if FRAMES else range(scn.frame_start, scn.frame_end + 1)
for f in frames:
    scn.frame_set(f); scn.render.filepath = os.path.join(OUT, f"{f:03d}.jpg"); bpy.ops.render.render(write_still=True)
