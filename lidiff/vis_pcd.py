import open3d as o3d
import numpy as np
import click
        
                                                                                                                                     
@click.command()
@click.option('--path', '-p', type=str, default='/data0/code/LiDiff-main/lidiff/results/pcd_full.ply', help='path to pcd')
@click.option('--radius', '-r', type=float, default=50., help='range to filter pcd')
def main(path, radius):
    pcd = o3d.io.read_point_cloud(path)

    if radius > 0.:
        points = np.array(pcd.points)
        dist = np.sum(points**2, -1)**0.5
        pcd.points = o3d.utility.Vector3dVector(points[(dist < radius) & (points[:,-1] < 3.) & (points[:,-1] > -2.5)])

    pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # 获取渲染选项并设置点的大小
    render_option = vis.get_render_option()
    render_option.point_size = 4.0
    
    # 可视化点云
    vis.run()
    vis.destroy_window()
                                                                                                                                     
if __name__ == '__main__':
    main()
