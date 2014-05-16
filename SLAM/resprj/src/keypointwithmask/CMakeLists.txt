ADD_EXECUTABLE(keypointwithmask keypointwithmask.cpp)
FIND_PACKAGE( OpenCV REQUIRED )
TARGET_LINK_LIBRARIES(keypointwithmask ${OpenCV_LIBS})
