/******************************************
 * Multi-ICP: We first extract indecies of each plane from pointcloud, then
 get the SIFT or FAST keypoint, and perform ICP on each class.
 ******************************************/

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

//CV
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <opencv2/core/eigen.hpp>

//PCL
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>

//g2o
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>

using namespace std;
using namespace cv;
using namespace g2o;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

typedef BlockSolver< BlockSolverTraits<-1, -1> >  SlamBlockSolver;
typedef LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

struct PLANE
{
    pcl::ModelCoefficients coff;  //a,b,c,d
    vector<KeyPoint> kp;             //关键点:图像位置
    vector<Point3f> kp_pos;        //关键点的3D位置
    Mat desp;                               //描述子
    Mat image;                             //图像
};

//////////////////////////////////////////////////
cv::Mat rgb1, rgb2, dep1, dep2; //data
PointCloud::Ptr cloud1, cloud2;
string dataPath = "/home/y/code/dataset/slam/";
vector<PLANE> planes1, planes2;

//Parameters
double distance_threshold = 0.05; //error of each plane model
double percent = 0.3;   //percent of planes in each pointcloud
double match_min_dist = 30; //最大匹配距离
double camera_factor = 5000; //相机参数
double camera_fx = 517.0, camera_fy = 517.0, camera_cx = 318.6, camera_cy = 255.3;
double min_error_plane = 0.01;  //归类关键点时的最小误差
vector<DMatch> inlierMatches;
//////////////////////////////////////////////////
void usage()
{
    cout<<"multiICP frame1 frame2 [distance]"<<endl;
}

void readData(string file1, string file2)
{
    rgb1 = cv::imread(dataPath + string("rgb_index/") + file1 + string(".png"), CV_LOAD_IMAGE_GRAYSCALE);
    rgb2 = cv::imread(dataPath + string("rgb_index/") + file2 + string(".png"), CV_LOAD_IMAGE_GRAYSCALE);
    dep1 = cv::imread(dataPath + string("dep_index/") + file1 + string(".png"), cv::IMREAD_ANYDEPTH);
    dep2 = cv::imread(dataPath + string("dep_index/") + file2 + string(".png"), cv::IMREAD_ANYDEPTH);

    cloud1 = PointCloud::Ptr(new PointCloud);
    cloud2 = PointCloud::Ptr(new PointCloud);
    
    pcl::io::loadPCDFile(dataPath+string("pcd/")+file1+string(".pcd"), *cloud1);
    pcl::io::loadPCDFile(dataPath+string("pcd/")+file2+string(".pcd"), *cloud2);

    cout<<"read data ok."<<endl;
}

vector<PLANE> extractPlanes(PointCloud::Ptr cloud);  //从两个点云中分割平面
void generateImage(PLANE& plane, Mat& depth);                              //根据平面的参数生成图像蒙板
void compute3dPosition(PLANE& plane, Mat& depth);   
vector<KeyPoint> extractKeypoints(cv::Mat image);     //从图像中提取关键点，默认为SIFT
Mat extractDescriptor(Mat image, vector<KeyPoint>& kp);  //根据关键点提取特征
vector<DMatch> match(vector<PLANE>& planes1, vector<PLANE>& planes2);  //匹配两组平面，以法向量作为特征
vector<DMatch> match(Mat desp1, Mat desp2);  //匹配两组特征点
int classifyKeypoints(KeyPoint kp, vector<PLANE>& planes, Mat depth, Point3f& pos);  //将特征点根据不同的平面进行归类
Eigen::Isometry3d PnP(PLANE& p1, PLANE& p2);  //求解PnP问题
Eigen::Matrix4f PnPUsingICP( PLANE& p1, PLANE& p2); //用pcl中的ICP求解PnP问题
//////////////////////////////////////////////////
//坐标变换
inline Point3f g2o2cv(Point3f p)
{
    return Point3f(-p.y, -p.z, p.x);
}

inline Point3f cv2g2o(Point3f p)
{
    return Point3f(p.z, -p.x, -p.y);
}
////////////////////////////////////////
int main(int argc, char** argv)
{
    if (argc < 3)
    {
        usage();
        return -1;
    }
    if (argc == 4)
        distance_threshold = atof(argv[3]);
    
    readData(argv[1], argv[2]);

    //extract plane
    cout<<"planes of cloud1: "<<endl;
    planes1 = extractPlanes( cloud1 );
    for (size_t i = 0; i<planes1.size(); i++)
    {
        pcl::ModelCoefficients c = planes1[i].coff;
        cout<<"Model coefficients of plane "<<i<<": "
            <<c.values[0]<<", "<<c.values[1]<<", "<<c.values[2]<<", "<<c.values[3]<<endl;
        planes1[i].image = rgb1.clone();
    }

    cout<<"planes of cloud2: "<<endl;
    planes2 = extractPlanes( cloud2 );
    for (size_t i = 0; i<planes2.size(); i++)
    {
        pcl::ModelCoefficients c = planes2[i].coff;
        cout<<"Model coefficients of plane "<<i<<": "
            <<c.values[0]<<", "<<c.values[1]<<", "<<c.values[2]<<", "<<c.values[3]<<endl;
        planes2[i].image = rgb2.clone();
    }

    //match two planes，以两组平面的法向量为基本特征
    vector<DMatch> matches = match( planes1, planes2 );
    cout<<"Total Matches: "<<matches.size()<<endl;
    for (size_t i=0; i<matches.size(); i++)
    {
        cout<<matches[i].queryIdx<<" --- "<<matches[i].trainIdx<<endl; //query在前，train在后
    }

    //尝试生成各平面的二维图像供opencv分析
    for (size_t i=0; i<planes1.size(); i++)
    {
        generateImage(planes1[i], dep1);
        planes1[i].kp = extractKeypoints( planes1[i].image);
        Mat image_keypoints;
        drawKeypoints( planes1[i].image, planes1[i].kp, image_keypoints );
        
        imshow("keypoints", image_keypoints);
        waitKey(0);
    }

    for (size_t i=0; i<planes2.size(); i++)
    {
        generateImage( planes2[i], dep2 );
        planes2[i].kp = extractKeypoints( planes2[i].image);
    }
    
    //根据两组平面的匹配关系，对KeyPoint进行匹配
    //生成描述子    并 计算各关键点的3d位置
    for (size_t i = 0; i<planes1.size(); i++)
    {
        planes1[i].desp = extractDescriptor( rgb1, planes1[i].kp );
        compute3dPosition( planes1[i], dep1 );
    }
    for (size_t i = 0; i<planes2.size(); i++)
    {
        planes2[i].desp = extractDescriptor( rgb2, planes2[i].kp );
        compute3dPosition( planes2[i], dep2 );
    }
    
    //分别匹配两组平面上的特征点
    cout<<"Using RANSAC to compute Transform Matrix."<<endl;
    vector<Eigen::Isometry3d> transforms;
    for (size_t i=0; i<matches.size(); i++)
    {
        cout<<"solving plane1: "<<matches[i].queryIdx<<" with plane2: "<<matches[i].trainIdx<<endl;
        Eigen::Isometry3d t = PnP( planes1[matches[i].queryIdx], planes2[matches[i].trainIdx] );
        transforms.push_back(t);

        //Eigen::Matrix4f T = PnPUsingICP( planes1[matches[i].queryIdx], planes2[matches[i].trainIdx] );
        //cout<<"result of ICP: "<<endl<<T<<endl;
    }

    cout<<"Transforms: "<<endl;
    for (size_t i =0; i<transforms.size(); i++)
    {
        cout<<"T"<<i<<" = "<<endl;
        cout<<transforms[i].matrix()<<endl;
    }

    //构造g2o的图
    cout<<"init g2o"<<endl;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    OptimizationAlgorithmGaussNewton* solver = new OptimizationAlgorithmGaussNewton( blockSolver );
    SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver );
    VertexSE3* v1 = new VertexSE3();
    VertexSE3* v2 = new VertexSE3();
    v1->setId(0);    v2->setId(1);
    v1->setEstimate( Eigen::Isometry3d::Identity() );
    v2->setEstimate( Eigen::Isometry3d::Identity() );
    v1->setFixed( true );
    
    optimizer.addVertex( v1 );
    optimizer.addVertex( v2 );
    //加入边
    for (size_t i=0; i<transforms.size(); i++)
    {
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices() [0] = optimizer.vertex(0);
        edge->vertices() [1] = optimizer.vertex(1);
        edge->setMeasurement( transforms[i] );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(1,1) = information(2,2) = 100; //位置误差
        information(3,3) = information(4,4) = information(5,5) = 100; //姿态误差
        edge->setInformation( information );
        optimizer.addEdge( edge );
        
    }

    //开始优化
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize( 10 );

    optimizer.save("./data/final.g2o");
    VertexSE3* final = dynamic_cast<VertexSE3*> (optimizer.vertex(1));
    Eigen::Isometry3d esti = final->estimate();
    cout<<"after g2o: "<<endl;
    cout<<esti.matrix()<<endl;
    
    return 0;
}
////////////////////////////////////////
vector<PLANE> extractPlanes(PointCloud::Ptr cloud)
{
    vector<PLANE> planes;
    pcl::ModelCoefficients::Ptr coefficients( new pcl::ModelCoefficients() );
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices );

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients( true );

    seg.setModelType( pcl::SACMODEL_PLANE );
    seg.setMethodType( pcl::SAC_RANSAC );
    seg.setDistanceThreshold( distance_threshold );

    int n = cloud->points.size();
    int i=0;

    PointCloud::Ptr tmp (new PointCloud());
    pcl::copyPointCloud(*cloud, *tmp);
    
    while( tmp->points.size() > percent*n )
    {
        seg.setInputCloud(tmp);
        seg.segment( *inliers, *coefficients );
        if (inliers->indices.size() == 0)
            break;
        PLANE p;
        p.coff = *coefficients;
        
        if ( coefficients->values[3] < 0)
        {
            for (int i=0; i<4; i++)
                p.coff.values[i] = -p.coff.values[i];
        }
        
        planes.push_back(p);

        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud( tmp );
        extract.setIndices( inliers );
        extract.setNegative( true );
        extract.filter( *tmp );
    }

    return planes;
}

vector<DMatch> match(vector<PLANE>& planes1, vector<PLANE>& planes2)
{
    //    FlannBasedMatcher matcher;
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    cv::Mat des1(planes1.size(), 4, CV_32F), des2(planes2.size(), 4, CV_32F);
    cout<<"start matching two planes"<<endl;
    for (size_t i=0; i<planes1.size(); i++)
    {
        pcl::ModelCoefficients c = planes1[i].coff;
        float m[1][4] = { c.values[0], c.values[1], c.values[2], c.values[3] };
        Mat mat = Mat(1,4, CV_32F, m);
        mat.row(0).copyTo( des1.row(i) );
    }

    for (size_t i=0; i<planes2.size(); i++)
    {
        pcl::ModelCoefficients c = planes2[i].coff;
        float m[1][4] = { c.values[0], c.values[1], c.values[2], c.values[3] };
        Mat mat = Mat(1,4, CV_32F, m);
        mat.row(0).copyTo( des2.row(i) );
    }

    cout<<"des1="<<des1<<endl<<"des2="<<des2<<endl;
    matcher.match( des1, des2, matches);
    double max_dist = 0, min_dist = 100;
    for (int i=0; i<des1.rows; i++)
    {
        double dist = matches[ i ].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    cout<<"match: min dist = "<<min_dist<<endl;
    //choose good matches
    vector<DMatch> good_matches;
    
    for (int i=0; i<des1.rows; i++)
    {
        if (matches[ i ].distance <= max(2*min_dist, match_min_dist))
        {
            good_matches.push_back(matches[ i ]);
        }
    }
    return good_matches;
}
vector<DMatch> match( Mat desp1, Mat desp2 )
{
    FlannBasedMatcher matcher;
    vector<DMatch> matches;

    if (desp1.empty() || desp2.empty())
    {
        return matches;
    }
    
    matcher.match( desp1, desp2, matches);
    double max_dist = 0, min_dist = 100;
    for (size_t i=0; i<matches.size(); i++)
    {
        double dist = matches[ i ].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    //choose good matches
    vector<DMatch> good_matches;
    
    for (size_t i=0; i<matches.size(); i++)
    {
        if (matches[ i ].distance <= max(2*min_dist, match_min_dist))
        {
            good_matches.push_back(matches[ i ]);
        }
    }

    return good_matches;
}
vector<KeyPoint> extractKeypoints(Mat image)
{
    initModule_nonfree();
    Ptr<FeatureDetector> detector = FeatureDetector::create("STAR");
    vector<KeyPoint> kp;
    detector->detect(image, kp);

    return kp;
}

int classifyKeypoints(KeyPoint kp, vector<PLANE>& planes, Mat depth, Point3f& pos)
{
    double u = kp.pt.x, v = kp.pt.y;
    unsigned short d = depth.at<unsigned short>(round(v), round(u));
    if (d == 0)
        return -1;
    double x = double(d)/camera_factor;
    double y = -( u - camera_cx) * x / camera_fx;
    double z = -( v - camera_cy) * x / camera_fy;

    pos = Point3f(x,y,z);
    vector<double> error;
    for (size_t i=0; i<planes.size(); i++)
    {
        pcl::ModelCoefficients c =planes[i].coff;
        double e = c.values[0]*x + c.values[1]*y + c.values[2]*z + c.values[3];
        e *= e;
        error.push_back(e);
    }

    double min_error = 9999;
    int min_error_index = 0;
    for (size_t i=0; i<error.size(); i++)
    {
        if (error[i] < min_error)
        {
            min_error = error[i];
            min_error_index = i;
        }
    }

    if (min_error > min_error_plane)
        return -1;
    return min_error_index;
}

void compute3dPosition(PLANE& plane, Mat& depth)
{
    for (size_t i=0; i<plane.kp.size(); i++)
    {
        double u = plane.kp[i].pt.x, v = plane.kp[i].pt.y;
        unsigned short d = depth.at<unsigned short>(round(v), round(u));
        if (d == 0)
        {
            plane.kp_pos.push_back( Point3f(0,0,0) );
            continue;
        }
        double x = double(d)/camera_factor;
        double y = -( u - camera_cx) * x / camera_fx;
        double z = -( v - camera_cy) * x / camera_fy;
        plane.kp_pos.push_back(Point3f( x, y, z) );
    }
}

Mat extractDescriptor( Mat image, vector<KeyPoint>& kp)
{
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );
    Mat descriptors;
    descriptor_extractor->compute(image, kp, descriptors);

    return descriptors;
}

Eigen::Isometry3d PnP(PLANE& p1, PLANE& p2)
{
    vector<DMatch> matches = match(p1.desp, p2.desp);
    cout<<"good matches: "<<matches.size()<<endl;
    //这是用cv的solvePnPRANSAC
    vector<Point3f> obj;  //目标点（cv坐标系下）
    vector<Point2f> img; //图像点（keypoints）
    for (size_t i=0; i<matches.size(); i++)
    {
        obj.push_back( g2o2cv(p1.kp_pos[matches[i].queryIdx]) );
        img.push_back( p2.kp[matches[i].trainIdx].pt );
    }
    double camera_matrix[3][3] = { { camera_fx, 0, camera_cx }, { 0, camera_fy ,camera_cy }, { 0, 0, 1 }};
    Mat cameraMatrix(3,3,CV_64F, camera_matrix);

    Mat rvec, tvec; 

    Mat inliers;     
    solvePnPRansac(obj, img, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers);
    inlierMatches.clear();
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( matches[inliers.at<int>(i,0)] );
    
    cout<<"inliers = "<<inliers.rows<<endl;
    if (inliers.rows < 5)
    {
        cerr<<"No enough inliers."<<endl;
    }
    //画出匹配结果
    Mat image_matches;
    drawMatches(rgb1, p1.kp, rgb2, p2.kp, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
    imshow("match", image_matches);
    waitKey(0);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    // 旋转向量转换成旋转矩阵
    Mat R;
    Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    cv2eigen(R, r);

    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = angle;
    
    return T;
}

Eigen::Matrix4f PnPUsingICP(PLANE& p1, PLANE& p2)
{
    vector<DMatch>& matches = inlierMatches;
    //构造待匹配的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>()), tgt(new pcl::PointCloud<pcl::PointXYZ>());
    for (size_t i=0; i<matches.size(); i++)
    {
        Point3f point1 = p1.kp_pos[matches[i].queryIdx];
        src->push_back( pcl::PointXYZ(point1.x, point1.y, point1.z) );
        Point3f point2 = p2.kp_pos[ matches[i].trainIdx ];
        tgt->push_back( pcl::PointXYZ(point2.x, point2.y, point2.z));
    }

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    icp.setInputSource( src );
    icp.setInputTarget( tgt );

    pcl::io::savePCDFile( "./data/src.pcd", *src );
    pcl::io::savePCDFile( "./data/tgt.pcd", *tgt );
    cout<<"source and target pointcloud is saved."<<endl;
    //参数
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    icp.setMaxCorrespondenceDistance (0.05);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations (50);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon (1e-8);
    // Set the euclidean distance difference epsilon (criterion 3)
    icp.setEuclideanFitnessEpsilon (1);

    pcl::PointCloud<pcl::PointXYZ>final;
    icp.align(final);

    cout<<"icp has converged: "<<icp.hasConverged()<<endl;
    Eigen::Matrix4f T = icp.getFinalTransformation(); //T是从src到tgt的变换

    if (icp.hasConverged())
    {
        //输出结果
        PointCloud::Ptr output( new PointCloud() );
        pcl::transformPointCloud( *cloud1, *output, T );
        *output += *cloud2;
        pcl::io::savePCDFile("./data/final.pcd", *output);
        cout<<"result is saved. Please check ./data/final.pcd "<<endl;
        waitKey(0);
    }
    return T;

}

void generateImage( PLANE& plane, Mat& depth )
{
    for( int i=0; i<plane.image.cols; i++)
        for (int j=0; j<plane.image.rows; j++)
        {
            unsigned short d = depth.at<unsigned short>(j, i);
            if (d == 0)
            {
                plane.image.at<unsigned char>( j, i) = 0;
                continue;
            }

            //看此点是否位于平面上
            double x = double(d)/camera_factor;
            double y = -( i - camera_cx) * x / camera_fx;
            double z = -( j - camera_cy) * x / camera_fy;
            
            double e = plane.coff.values[0]*x + plane.coff.values[1]*y + plane.coff.values[2]*z + plane.coff.values[3];
            e*=e;
            if ( e<0.02 )
            {
                //认为该点位于此平面
            }
            else
                plane.image.at<unsigned char> (j, i) = 0;
        }

    //将图像做一次直方图均衡化
    Mat dst;
    equalizeHist( plane.image, dst );
    plane.image = dst.clone();
}
