"""
Step 1 (MeshGraphNets 鐗?: 浠?meshgraphnets TFRecord 鏁版嵁涓彁鍙栧渾鏌卞熬娴佸尯
       鏃堕棿骞冲潎閫熷害鍦猴紝鎻掑€煎埌瑙勫垯缃戞牸鍚庤緭鍑?CSV銆?
鏁版嵁婧? D:\keyan\cylinder_data\meshgraphnets\train.tfrecord
  - TFRecord 涓墍鏈夊瓧娈靛潎瀛樺偍涓?bytes_list (搴忓垪鍖?numpy 浜岃繘鍒?
  - 浣跨敤 google.protobuf 瑙ｆ瀽 + np.frombuffer 瑙ｇ爜
杈撳嚭鏍煎紡涓庡師 step1_extract_wake.py 瀹屽叏涓€鑷达紝渚?step2/step3 鐩存帴浣跨敤銆?"""
import numpy as np
import pandas as pd
import json
import os
import struct
from scipy.interpolate import griddata
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

# 鈹€鈹€ 璺緞 鈹€鈹€
TFRECORD_PATH = r"D:\keyan\cylinder_data\meshgraphnets\train.tfrecord"
META_PATH     = r"D:\keyan\cylinder_data\meshgraphnets\meta.json"
OUT_DIR       = r"D:\keyan\cylinder_flow_pisr\outputs_mgn"
os.makedirs(OUT_DIR, exist_ok=True)

# 鈹€鈹€ 1. 璇诲彇 meta.json 鈹€鈹€
with open(META_PATH) as f:
    meta_raw = json.load(f)

traj_len = meta_raw["trajectory_length"]  # 600
print(f"Trajectory length: {traj_len}")


# 鈹€鈹€ 2. 鏋勫缓 protobuf Example 娑堟伅绫诲瀷 鈹€鈹€
def build_example_class():
    """鐢?google.protobuf 鍔ㄦ€佹瀯寤?tf.train.Example 娑堟伅绫诲瀷銆?"""
    fp = descriptor_pb2.FileDescriptorProto()
    fp.name = "example.proto"
    fp.syntax = "proto3"

    # BytesList { repeated bytes value = 1; }
    m = fp.message_type.add(); m.name = "BytesList"
    f = m.field.add(); f.name = "value"; f.number = 1; f.type = 12; f.label = 3

    # FloatList { repeated float value = 1 [packed=true]; }
    m = fp.message_type.add(); m.name = "FloatList"
    f = m.field.add(); f.name = "value"; f.number = 1; f.type = 2; f.label = 3
    f.options.CopyFrom(descriptor_pb2.FieldOptions()); f.options.packed = True

    # Int64List { repeated int64 value = 1 [packed=true]; }
    m = fp.message_type.add(); m.name = "Int64List"
    f = m.field.add(); f.name = "value"; f.number = 1; f.type = 3; f.label = 3
    f.options.CopyFrom(descriptor_pb2.FieldOptions()); f.options.packed = True

    # Feature { oneof kind { BytesList=1; FloatList=2; Int64List=3; } }
    m = fp.message_type.add(); m.name = "Feature"
    oneof = m.oneof_decl.add(); oneof.name = "kind"
    for fname, fnum, tname in [("bytes_list", 1, "BytesList"),
                                ("float_list", 2, "FloatList"),
                                ("int64_list", 3, "Int64List")]:
        f = m.field.add()
        f.name = fname; f.number = fnum; f.type = 11; f.label = 1
        f.type_name = tname; f.oneof_index = 0

    # Features { map<string, Feature> feature = 1; }
    m = fp.message_type.add(); m.name = "Features"
    n = m.nested_type.add(); n.name = "FeatureEntry"
    n.options.CopyFrom(descriptor_pb2.MessageOptions()); n.options.map_entry = True
    f = n.field.add(); f.name = "key"; f.number = 1; f.type = 9; f.label = 1
    f = n.field.add(); f.name = "value"; f.number = 2; f.type = 11; f.label = 1; f.type_name = "Feature"
    f = m.field.add(); f.name = "feature"; f.number = 1; f.type = 11; f.label = 3; f.type_name = "FeatureEntry"

    # Example { Features features = 1; }
    m = fp.message_type.add(); m.name = "Example"
    f = m.field.add(); f.name = "features"; f.number = 1; f.type = 11; f.label = 1; f.type_name = "Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fp)
    factory = message_factory.MessageFactory(pool=pool)
    return factory.GetPrototype(pool.FindMessageTypeByName("Example"))


def read_first_tfrecord(path):
    """浠?TFRecord 鏂囦欢璇诲彇绗竴鏉¤褰曘€?"""
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        f.read(4)  # crc
        data = f.read(length)
        f.read(4)  # crc
        return data


def parse_tfrecord(path, meta):
    """瑙ｆ瀽 TFRecord锛岃繑鍥?{field_name: numpy_array} 瀛楀吀銆?"""
    ExampleClass = build_example_class()
    raw = read_first_tfrecord(path)
    print(f"  Record size: {len(raw) / 1024 / 1024:.1f} MB")

    example = ExampleClass()
    example.ParseFromString(raw)

    result = {}
    for key, info in meta["features"].items():
        feature = example.features.feature[key]
        kind = feature.WhichOneof("kind")

        np_dtype = np.float32 if info["dtype"] == "float32" else np.int32

        if kind == "bytes_list":
            # 鏁版嵁浠ュ簭鍒楀寲浜岃繘鍒?blob 瀛樺偍
            raw_bytes = feature.bytes_list.value[0]
            result[key] = np.frombuffer(raw_bytes, dtype=np_dtype).copy()
        elif kind == "float_list":
            result[key] = np.array(feature.float_list.value, dtype=np.float32)
        elif kind == "int64_list":
            result[key] = np.array(feature.int64_list.value, dtype=np_dtype)
        else:
            raise ValueError(f"Unexpected feature kind for {key}: {kind}")

        print(f"  {key}: {kind}, dtype={np_dtype}, len={len(result[key])}")

    return result


# 鈹€鈹€ 3. 瑙ｆ瀽鏁版嵁 鈹€鈹€
print("Parsing TFRecord (first trajectory) ...")
raw_data = parse_tfrecord(TFRECORD_PATH, meta_raw)

# 鈹€鈹€ 4. 閲嶅缓鏁版嵁褰㈢姸 鈹€鈹€
mesh_pos = raw_data["mesh_pos"].reshape(-1, 2)        # (N_nodes, 2)
n_nodes = mesh_pos.shape[0]
cells = raw_data["cells"].reshape(-1, 3)               # (N_cells, 3)
node_type = raw_data["node_type"][:n_nodes]             # (N_nodes,)
velocity = raw_data["velocity"].reshape(traj_len, n_nodes, 2)
pressure = raw_data["pressure"].reshape(traj_len, n_nodes, 1)

print(f"\nNodes: {n_nodes}, Cells: {cells.shape[0]}")
print(f"mesh_pos: x=[{mesh_pos[:,0].min():.4f}, {mesh_pos[:,0].max():.4f}], "
      f"y=[{mesh_pos[:,1].min():.4f}, {mesh_pos[:,1].max():.4f}]")
print(f"velocity shape: {velocity.shape}")

# 鈹€鈹€ 5. 璇嗗埆鑺傜偣绫诲瀷 鈹€鈹€
# MeshGraphNets: 0=NORMAL, 4=INLET, 5=OUTLET, 6=WALL_BOUNDARY
unique_types = np.unique(node_type)
print(f"Node types: {unique_types}")
for nt in unique_types:
    mask = (node_type == nt)
    pts = mesh_pos[mask]
    print(f"  type {nt}: {mask.sum()} nodes, "
          f"x=[{pts[:,0].min():.4f},{pts[:,0].max():.4f}], "
          f"y=[{pts[:,1].min():.4f},{pts[:,1].max():.4f}]")

# 鈹€鈹€ 6. 妫€娴嬪渾鏌?(閫氳繃缃戞牸绌烘礊 + 杩戝浣庨€熷尯鍩? 鈹€鈹€
# meshgraphnets cylinder_flow: 闈炵粨鏋勫寲涓夎缃戞牸锛屽渾鏌变綋鐜颁负缃戞牸涓殑"娲?
# 绛栫暐: 鎵惧埌缃戞牸瀵嗗害鏈€楂樼殑鐜姸鍖哄煙 (鍦嗘煴琛ㄩ潰闄勮繎缃戞牸鏈€瀵?
t_start = traj_len // 2
u_mean_all = velocity[t_start:, :, 0].mean(axis=0)
v_mean_all = velocity[t_start:, :, 1].mean(axis=0)
speed = np.sqrt(u_mean_all**2 + v_mean_all**2)

from scipy.spatial import cKDTree
interior = (node_type == 0)

# 鏂规硶: 鎵惧埌鏈€杩戦偦璺濈鏈€灏忕殑鍖哄煙 (缃戞牸鏈€瀵?= 鍦嗘煴闄勮繎)
tree = cKDTree(mesh_pos[interior])
dd, _ = tree.query(mesh_pos[interior], k=4)
avg_nn_dist = dd[:, 1:].mean(axis=1)

# 鍙栫綉鏍兼渶瀵嗙殑 3% 鑺傜偣
dense_thresh = np.percentile(avg_nn_dist, 3)
dense_mask_local = avg_nn_dist < dense_thresh
dense_pts = mesh_pos[interior][dense_mask_local]

# 杩欎簺瀵嗛泦鐐瑰簲鐜粫鍦嗘煴锛岀敤瀹冧滑浼拌鍦嗘煴涓績
cyl_cx = dense_pts[:, 0].mean()
cyl_cy = dense_pts[:, 1].mean()

# 浠庡瘑闆嗙偣鍒颁腑蹇冪殑璺濈鍒嗗竷鏉ヤ及绠楀崐寰?dist_to_center = np.sqrt((dense_pts[:, 0] - cyl_cx)**2 + (dense_pts[:, 1] - cyl_cy)**2)
# 鍙栦腑浣嶆暟浣滀负鍗婂緞 (姣?mean/max 鏇撮瞾妫?
radius = np.median(dist_to_center)
D = 2 * radius

print(f"\nCylinder detected via mesh density:")
print(f"  Center: ({cyl_cx:.4f}, {cyl_cy:.4f})")
print(f"  Estimated R={radius:.4f}, D={D:.4f}")
print(f"  Dense nodes used: {dense_mask_local.sum()}")

# 鍚堢悊鎬ф鏌? DFG benchmark 閫氬父 D=0.1, domain 1.6x0.41
domain_w = mesh_pos[:, 0].max() - mesh_pos[:, 0].min()
if D > 0.3 * domain_w or D < 0.01 * domain_w:
    print(f"  WARNING: D={D:.4f} seems unreasonable for domain width {domain_w:.4f}")
    print(f"  Falling back to DFG benchmark: D=0.1, center=(0.2, 0.2)")
    cyl_cx, cyl_cy = 0.2, 0.2
    D = 0.1
    radius = 0.05

# 鈹€鈹€ 7. 纭畾鍏ュ彛閫熷害 鈹€鈹€
# type=4 鍦ㄨ繖涓暟鎹泦涓槸 x=0 鐨勫叆鍙ｈ妭鐐?inlet_mask = (node_type == 4)
if inlet_mask.sum() == 0:
    x_threshold = mesh_pos[:, 0].min() + 0.02
    inlet_mask = mesh_pos[:, 0] < x_threshold

u_inlet_profile = velocity[t_start:, inlet_mask, 0].mean(axis=0)  # 鏃堕棿骞冲潎鍏ュ彛u鍒嗗竷
U_in_mean = u_inlet_profile.mean()  # 骞冲潎鍏ュ彛閫熷害
U_in_max = u_inlet_profile.max()    # 鏈€澶у叆鍙ｉ€熷害 (鎶涚墿绾垮嘲鍊?
# 瀵逛簬 PISR锛岀敤骞冲潎鍏ュ彛閫熷害鍋氭棤閲忕翰鍖栨洿鍚堢悊
U_in = U_in_mean if U_in_mean > 0.01 else 1.0
print(f"\nInlet velocity: U_mean={U_in_mean:.4f}, U_max={U_in_max:.4f}")
print(f"Using U_in={U_in:.4f} for normalization")

# 鈹€鈹€ 8. 鏃堕棿骞冲潎閫熷害鍦?鈹€鈹€
print(f"\nTime-averaging over steps {t_start}..{traj_len} ...")
u_mean_nodes = u_mean_all  # 宸插湪涓婇潰璁＄畻
v_mean_nodes = v_mean_all
print(f"u_mean range: [{u_mean_nodes.min():.4f}, {u_mean_nodes.max():.4f}]")
print(f"v_mean range: [{v_mean_nodes.min():.4f}, {v_mean_nodes.max():.4f}]")

# 鈹€鈹€ 9. 鎻掑€煎埌瑙勫垯缃戞牸 鈹€鈹€
# 鍦嗘煴涓嬫父灏炬祦鍖? 浠庡渾鏌变腑蹇冨線涓嬫父 2D ~ 8D
wake_x_lo = cyl_cx + 2 * D
wake_x_hi = cyl_cx + 8 * D
wake_y_lo = cyl_cy - 3 * D
wake_y_hi = cyl_cy + 3 * D

# 瑁佸壀鍒板煙杈圭晫 (鐣?margin 閬垮厤杈圭晫鏁堝簲)
margin = 0.01
x_domain_min = mesh_pos[:, 0].min() + margin
x_domain_max = mesh_pos[:, 0].max() - margin
y_domain_min = mesh_pos[:, 1].min() + margin
y_domain_max = mesh_pos[:, 1].max() - margin
wake_x_lo = max(wake_x_lo, x_domain_min)
wake_x_hi = min(wake_x_hi, x_domain_max)
wake_y_lo = max(wake_y_lo, y_domain_min)
wake_y_hi = min(wake_y_hi, y_domain_max)

print(f"\nWake interpolation region:")
print(f"  x: [{wake_x_lo:.4f}, {wake_x_hi:.4f}]")
print(f"  y: [{wake_y_lo:.4f}, {wake_y_hi:.4f}]")
print(f"  x/D: [{(wake_x_lo-cyl_cx)/D:.1f}, {(wake_x_hi-cyl_cx)/D:.1f}]")
print(f"  y/D: [{(wake_y_lo-cyl_cy)/D:.1f}, {(wake_y_hi-cyl_cy)/D:.1f}]")

n_x_grid = 48  # x 鏂瑰悜閲囨牱鐐规暟 (瑕嗙洊鏇村鑼冨洿)
n_y_grid = 64  # y 鏂瑰悜閲囨牱鐐规暟

x_grid = np.linspace(wake_x_lo, wake_x_hi, n_x_grid)
y_grid = np.linspace(wake_y_lo, wake_y_hi, n_y_grid)
dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]

XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
grid_points = np.column_stack([XX.ravel(), YY.ravel()])

print(f"Regular grid: {n_x_grid} x {n_y_grid} = {n_x_grid * n_y_grid} points")
print(f"  dx={dx:.6f}, dy={dy:.6f}")

print("Interpolating to regular grid ...")
u_mean_grid = griddata(mesh_pos, u_mean_nodes, grid_points, method="linear").reshape(n_x_grid, n_y_grid)
v_mean_grid = griddata(mesh_pos, v_mean_nodes, grid_points, method="linear").reshape(n_x_grid, n_y_grid)

# 澶勭悊 NaN
nan_count = np.isnan(u_mean_grid).sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN values, filling with nearest neighbor")
    u_near = griddata(mesh_pos, u_mean_nodes, grid_points, method="nearest").reshape(n_x_grid, n_y_grid)
    v_near = griddata(mesh_pos, v_mean_nodes, grid_points, method="nearest").reshape(n_x_grid, n_y_grid)
    nan_mask = np.isnan(u_mean_grid)
    u_mean_grid[nan_mask] = u_near[nan_mask]
    nan_mask_v = np.isnan(v_mean_grid)
    v_mean_grid[nan_mask_v] = v_near[nan_mask_v]

# 鈹€鈹€ 10. 閫夊彇绗﹀彿鍥炲綊鐢ㄥ熬娴佸瓙鍖哄煙 (3D ~ 5D) 鈹€鈹€
wake_sr_x_lo = cyl_cx + 3 * D
wake_sr_x_hi = cyl_cx + 5 * D
ix_sr = np.where((x_grid >= wake_sr_x_lo) & (x_grid <= wake_sr_x_hi))[0]

if len(ix_sr) < 3:
    print("WARNING: SR region too small, expanding range")
    wake_sr_x_lo = cyl_cx + 2 * D
    wake_sr_x_hi = cyl_cx + 6 * D
    ix_sr = np.where((x_grid >= wake_sr_x_lo) & (x_grid <= wake_sr_x_hi))[0]
if len(ix_sr) < 3:
    ix_sr = np.arange(n_x_grid // 4, 3 * n_x_grid // 4)

n_x_wake = len(ix_sr)
print(f"\nSR wake sub-region: {n_x_wake} x-positions, indices {ix_sr[0]}..{ix_sr[-1]}")
print(f"  x/D range: [{(x_grid[ix_sr[0]]-cyl_cx)/D:.2f}, {(x_grid[ix_sr[-1]]-cyl_cx)/D:.2f}]")

# 鈹€鈹€ 11. 鏋勫缓 DataFrame 鈹€鈹€
rows = []
for i in ix_sr:
    for j in range(n_y_grid):
        rows.append({
            "x_D": (x_grid[i] - cyl_cx) / D,
            "y_D": (y_grid[j] - cyl_cy) / D,
            "u_star": u_mean_grid[i, j] / U_in,
        })

df = pd.DataFrame(rows)
print(f"\nDataFrame: {len(df)} samples")
print(df.describe())

# v_mean 鍦ㄥ熬娴佸尯 (渚涙暎搴︽楠?
v_wake = v_mean_grid[np.ix_(ix_sr, range(n_y_grid))]

# 鈹€鈹€ 12. 淇濆瓨 鈹€鈹€
csv_path = os.path.join(OUT_DIR, "wake_data.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

np.save(os.path.join(OUT_DIR, "v_mean_wake.npy"), v_wake / U_in)
np.save(os.path.join(OUT_DIR, "wake_x.npy"), x_grid[ix_sr])
np.save(os.path.join(OUT_DIR, "wake_y.npy"), y_grid)

meta_out = {
    "D": float(D),
    "U_in": float(U_in),
    "Re": "estimated ~100 (DFG benchmark, not provided in data)",
    "cyl_cx": float(cyl_cx),
    "cyl_cy": float(cyl_cy),
    "dx": float(dx),
    "dy": float(dy),
    "wake_ix": ix_sr.tolist(),
    "n_x_wake": n_x_wake,
    "n_y": n_y_grid,
    "data_source": "meshgraphnets",
    "tfrecord": TFRECORD_PATH,
    "t_avg_start": t_start,
    "t_avg_end": traj_len,
    "n_nodes_original": n_nodes,
    "interp_method": "linear",
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta_out, f, indent=2)

np.save(os.path.join(OUT_DIR, "u_mean_full_grid.npy"), u_mean_grid / U_in)
np.save(os.path.join(OUT_DIR, "v_mean_full_grid.npy"), v_mean_grid / U_in)
np.save(os.path.join(OUT_DIR, "x_grid.npy"), x_grid)
np.save(os.path.join(OUT_DIR, "y_grid.npy"), y_grid)

print(f"\nAll outputs saved to {OUT_DIR}")
print("Done.")
