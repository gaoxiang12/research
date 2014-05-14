#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/common/io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>

using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointWithScale KeypointT;
typedef pcl::PointCloud<PointT> PointCloud;

const float min_scale = 0.01f;
const int n_octaves = 6;
const int n_scales_per_octave = 4;
const float min_contrast = 0.05f;

void usage()
{
    cout<<"sift3d filepath"<<endl;
}

int main(int argc, char** argv )
{
    if (argc < 2)
    {
        usage();
        return -1;
    }

    string pcdfile(argv[1]);
    PointCloud::Ptr cloud( new PointCloud() );
    pcl::io::loadPCDFile( pcdfile, *cloud );

    cout<<"original cloud size = "<<cloud->points.size()<<endl;
    pcl::PointCloud<KeypointT>::Ptr keypoints(new pcl::PointCloud<KeypointT>());
    pcl::SIFTKeypoint<PointT, KeypointT> sift;
    pcl::PointCloud<KeypointT> result;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT> ());

    sift.setSearchMethod( tree );
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud( cloud );
    sift.compute( result );

    cout<<"result size = "<<result.points.size()<<endl;
    // Copying the pointwithscale to pointxyz so as visualize the cloud
    PointCloud::Ptr cloud_temp (new PointCloud());
    copyPointCloud(result, *cloud_temp);
    std::cout << "SIFT points in the result are " << cloud_temp->points.size () << std::endl;
    // Visualization of keypoints along with the original cloud
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    pcl::visualization::PointCloudColorHandlerCustom<PointT> keypoints_color_handler (cloud_temp, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_color_handler (cloud, 255, 0, 0);
    viewer.setBackgroundColor( 0.0, 0.0, 0.0 );
    viewer.addPointCloud(cloud, cloud_color_handler, "cloud");
    viewer.addPointCloud(cloud_temp, keypoints_color_handler, "keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
  
    while(!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }
    
    return 0;
}
