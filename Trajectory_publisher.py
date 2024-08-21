#!/usr/bin/env python
import rospy
import pandas as pd
from std_msgs.msg import Header
from hmcl_msgs.msg import LaneArray, Lane, Waypoint
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

class LanePublisher:
    def __init__(self):
        rospy.init_node('lane_publisher')
        self.publisher = rospy.Publisher('/global_traj', LaneArray, queue_size=10)
        self.rate = rospy.Rate(1)  # 1Hz

        # CSV 파일 읽기
        self.cx, self.cy = self.read_csv_data('~/Downloads/very_dense_track_coordinates.csv')

    def read_csv_data(self, file_path):
        data = pd.read_csv(file_path)
        cx = data['x'].tolist()
        cy = data['y'].tolist()
        return cx, cy

    def build_lane_array(self):
        header = Header(stamp=rospy.Time.now(), frame_id="world")
        lanes = []
        lane = Lane()
        lane.waypoints = [] 

        # CSV에서 읽어온 좌표로 웨이포인트 생성
        for i in range(len(self.cx)):
            waypoint = Waypoint()
            waypoint.gid = i
            waypoint.lid = 100 + i  # 각 웨이포인트에 고유의 ID 부여
            waypoint.pose = PoseStamped(
                header=header,
                pose=Pose(
                    position=Point(x=self.cx[i], y=self.cy[i], z=0.0),
                    orientation=Quaternion(x=0, y=0, z=0, w=1)
                )
            )
            waypoint.lane_id = 0  # 모든 웨이포인트가 같은 차선에 속한다고 가정
            lane.waypoints.append(waypoint)
        
        lanes.append(lane)
        lane_array_msg = LaneArray(header=header, id=123, lanes=lanes)
        return lane_array_msg

    def publish_data(self):
        while not rospy.is_shutdown():
            lane_array_msg = self.build_lane_array()
            self.publisher.publish(lane_array_msg)
            rospy.loginfo("Published lane data")
            self.rate.sleep()

if __name__ == '__main__':
    lane_publisher = LanePublisher()
    lane_publisher.publish_data()