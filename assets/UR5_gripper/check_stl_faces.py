import os
import trimesh

# 设置 mesh 路径
mesh_dir = os.path.join(os.path.dirname(__file__), 'mesh', 'visual')
max_faces = 200000

# 递归查找 STL 文件
stl_files = [os.path.join(mesh_dir, f)
             for f in os.listdir(mesh_dir)
             if f.lower().endswith('.stl')]

print(f"🔍 在 {mesh_dir} 中检测到 {len(stl_files)} 个 STL 文件...\n")

# 检查每个 STL 的三角面数
for stl_path in stl_files:
    try:
        mesh = trimesh.load_mesh(stl_path)
        face_count = len(mesh.faces)
        if face_count > max_faces:
            print(f"❌ {os.path.basename(stl_path)} - 面数 {face_count} 超过 MuJoCo 限制")
        else:
            print(f"✅ {os.path.basename(stl_path)} - 面数 {face_count}")
    except Exception as e:
        print(f"⚠️ {os.path.basename(stl_path)} - 加载失败: {e}")

