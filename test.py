from plyfile import PlyData

with open("/home/kobe/point_cloud_rendering/data/47eb87b5bb/scans/pc_aligned.ply", 'rb') as f:
	print(PlyData.read(f))