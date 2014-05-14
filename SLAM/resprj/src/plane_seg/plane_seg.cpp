#include <iostream>
#include <string>
#include <sstream>
//#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
float planePercent;

void usage()
{
    cout<<"plane_seg filepath [plane\%]"<<endl;
}

int main( int argc, char** argv )
{
    if (argc < 2)
    {
        usage();
        return -1;
    }
    if (argc == 3)
        planePercent = atof(argv[2]);
    else
        planePercent = 0.3;
    string pcdfile(argv[1]);
    PointCloud::Ptr cloud( new PointCloud );
    pcl::io::loadPCDFile(pcdfile, *cloud);
    pcl::ModelCoefficients::Ptr coefficients( new pcl::ModelCoefficients );
    pcl::PointIndices::Ptr inliers( new pcl::PointIndices );

    pcl::SACSegmentation<PointT> seg;

    seg.setOptimizeCoefficients( true );

    seg.setModelType( pcl::SACMODEL_PLANE );
    seg.setMethodType( pcl::SAC_RANSAC );
    seg.setDistanceThreshold( 0.05 );

    int n = cloud->points.size();
    int i=1;
    while (cloud->points.size() > planePercent*n)
    {
        cout<<"loop "<<i<<" cloud size = "<<cloud->points.size()<<endl;
        seg.setInputCloud(cloud);
        seg.segment( *inliers, *coefficients );
        if (inliers->indices.size() == 0)
        {
            break;
        }
        
        //将抓取的平面分离出来
        PointCloud::Ptr p( new PointCloud );
    
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud( cloud );
        extract.setIndices( inliers );
        extract.setNegative(false);
        extract.filter(*p);
        string str;
        stringstream ss;
        ss<<"plane"<<i<<".pcd";
        pcl::io::savePCDFileASCII(ss.str(), *p);

        //剩下的重新放回原文件中
        extract.setNegative( true );
        extract.filter( *cloud );

        i++;
    }

    cout<<"total "<<i<<" planes."<<endl;
    return 0;
}

