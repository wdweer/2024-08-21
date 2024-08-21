import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
import math
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from tf.transformations import euler_from_quaternion
from cvxpy_mpc import MPC
from vehicle_model import VehicleModel
import utils
import numpy as np
from autoware_msgs.msg import DetectedObjectArray
from autoware_msgs.msg import DetectedObject
from hmcl_msgs.msg import LaneArray
import time
from hmcl_msgs.msg import Waypoint


class Model_Predictive():
    def __init__(self):
        rospy.init_node("Model_Predictive")
        self.target_pose_sub = rospy.Subscriber('/target_pose', Pose, self.target_pose_callback)
        self.target_velocity_sub = rospy.Subscriber('/target_velocity', Twist, self.target_velocity_callback)
        self.target_angular_sub = rospy.Subscriber('/target_angular', Float64, self.target_angular_callback)
        self.target_object_sub=rospy.Subscriber("target_objects", DetectedObjectArray, self.detected_object_callback)
        self.target_object_pub=rospy.Publisher("dynamic_pred_target_objects", DetectedObjectArray, self.detected_object_callback)
        self.global_traj_sub=rospy.Subscriber('/global_traj',LaneArray,self.global_traj_callback)
        self.overtaking_traj_pub=rospy.Publisher('/overtaking_traj',LaneArray, queue_size=1)
        self.model_predicted_num = 5
        self.dt = 0.1
        self.target_x = 0
        self.target_y = 0
        self.target_velocity = 0
        self.target_angular_z = 0
        self.control = np.zeros(2)
        self.state=[0]*4
        rate = rospy.Rate(1000)
        self.model_prediction_x = []
        self.model_prediction_y = []
        self.repulsed_potential_field_point_x=[]
        self.repulsed_potential_field_point_y=[]
        self.bubble_param=1.3
        self.object_dic={}
        self.bubble_param=1.3
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'bo-')
        self.line_model, = self.ax.plot([], [], 'bo-')
        self.line_repulsed, = self.ax.plot([], [], 'bo-')
        self.line_Bezier, = self.ax.plot([], [], 'bo-')
        self.ax.set_xlim(min(self.cx) - 1, max(self.cx) + 1)
        self.ax.set_ylim(min(self.cy) - 1, max(self.cy) + 1)
        self.track_line, = self.ax.plot(self.cx, self.cy, 'r--')
        mode='simulation'
        if mode=='simulation':
            self.animation = FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, blit=True)
        if mode=='Detect':
            self.animation = FuncAnimation(self.fig, self.update_plot1, init_func=self.init_plot, blit=True)
        while not rospy.is_shutdown():
            rate.sleep()
            plt.pause(0.001)
    
        
            
            
    def global_traj_callback(self,data):
        self.cx=[]
        self.cy=[]
        for i in data.lanes:
            for j in i.waypoints:
                self.cx.append(j.pose.pose.position.x)
                self.cy.append(j.pose.pose.position.y)
        
        
        self.lane_data=data.lanes

            #set mode simulation or Detect(ROS bag)
        rate=rospy.Rate(1000)
        self.path=utils.compute_path_from_wp(self.cx,self.cy,0.05)
        # self.model_prediction_x = []
        # self.model_prediction_y = []
            
    
        
    def init_plot(self):
        self.line.set_data([], [])
        self.track_line.set_data(self.cx, self.cy)
        return [self.line, self.track_line]
    def update_plot(self, frame):
    # Ensure model_prediction_x and model_prediction_y have initial values
        if not hasattr(self, 'model_prediction_x'):
            self.model_prediction_x = []
        if not hasattr(self, 'model_prediction_y'):
            self.model_prediction_y = []

    # Ensure repulsed_potential_field_point_x and repulsed_potential_field_point_y have initial values
        if not hasattr(self, 'repulsed_potential_field_point_x'):
            self.repulsed_potential_field_point_x = []
        if not hasattr(self, 'repulsed_potential_field_point_y'):
            self.repulsed_potential_field_point_y = []

    # Append the target positions to the prediction lists
        self.model_prediction_x.append(self.target_x)
        self.model_prediction_y.append(self.target_y)

    # Update the line objects
        self.line_model.set_data(self.model_prediction_x, self.model_prediction_y)
        self.line_model.set_color('r')

        self.line_repulsed.set_data(self.repulsed_potential_field_point_x, self.repulsed_potential_field_point_y)
        self.line_repulsed.set_color('b')
            
        self.line_Bezier.set_data(self.B[0], self.B[1])
        self.line_Bezier.set_color('c')
        
        # self.local_line.set_data(self.local_cx, self.local_cx)
        # self.local_line.set_color('c')

        return [self.line_model, self.line_repulsed,self.line_Bezier,self.track_line]
    def update_plot1(self,frame):
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # List of colors
        c = 0
        for i in self.obj:
            self.x=self.target_veh_dic_x[i]+self.Points_x
            self.y=self.target_veh_dic_y[i]+self.Points_y
            self.line.set_data(self.x,self.y)
            color = colors[c]  # Cycle through colors if there are more objects than colors

            self.line.set_color(color)  # Set the color for the current line
            c = c + 1
        return [self.line, self.track_line]


    def target_pose_callback(self, data):
        # set prediction_policy MPC or heuristic
        self.Prediction_policy='heuristic'
        self.target_x = data.position.x
        self.target_y = data.position.y
        self.target_z = data.position.z
        self.target_orientation_x=data.orientation.x 
        self.target_orientation_y=data.orientation.y 
        self.target_orientation_z=data.orientation.z 
        self.target_orientation_w=data.orientation.w 
        self.target_orientation = [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]
        if self.Prediction_policy=='heuristic':
           self.heuristic_model_predictive()
           self.Intuitive_Artifical_potential_field_2()# Signal to update plot
        elif self.Prediction_policy=='MPC':
           self.MPC_model_predictive()
           self.Intuitive_Artifical_potential_field_2()
           

    def target_velocity_callback(self, data):
        self.target_velocity_x = data.linear.x
        self.target_velocity_y = data.linear.y
        self.target_velocity=((self.target_velocity_x)**2+(self.target_velocity_y)**2)**(1/2)
        self.target_yaw = math.atan2(self.target_velocity_y, self.target_velocity_x)

    def target_angular_callback(self, data):
        self.target_angular_z = data.data

    def get_yaw_from_orientation(self, x, y, z, w):
        euler = euler_from_quaternion([x, y, z, w])
        return euler[2] 

    def heuristic_model_predictive(self):
        model_prediction_x = self.target_x
        target_velocity_x = self.target_velocity_x
        target_yaw = self.target_yaw
        model_prediction_y = self.target_y
        target_velocity_y = self.target_velocity_y
        target_angular_z = self.target_angular_z
        self.model_prediction_x = []
        self.model_prediction_y = []
        for i in range(self.model_predicted_num):
            self.model_prediction_x.append(model_prediction_x)
            model_prediction_x += target_velocity_x
            target_yaw += target_angular_z
            target_velocity_x = self.target_velocity * math.cos(target_yaw)
        for j in range(self.model_predicted_num):
            self.model_prediction_y.append(model_prediction_y)
            model_prediction_y += target_velocity_y
            target_yaw += target_angular_z
            target_velocity_y = self.target_velocity * math.sin(target_yaw)

    def MPC_model_predictive(self):
        T=1 #Prediction Horizon in Time
        L=0.3 #Vehicle_Wheelbase
        Q = [20, 20, 10, 20]  # state error cost [x,y,v,yaw]
        Qf = [30, 30, 30, 30]  # state error cost at final timestep [x,y,v,yaw]
        R = [10, 10]  # input cost [acc ,steer]
        P = [10, 10]  # input rate of change cost [acc ,steer]
        mpc = MPC(VehicleModel(), T, self.dt, Q, Qf, R, P)
        # if len(self.model_prediction_x)>=10: 
        self.model_prediction_x = []
        self.model_prediction_y = []
        self.state[0]=self.target_x
        self.state[1]=self.target_y
        self.state[2]=self.target_velocity
        self.state[3]=self.target_yaw
        target = utils.get_ref_trajectory(self.state, self.path, self.target_velocity, T, self.dt)
        ego_state = np.array([0.0, 0.0, self.state[2], 0.0])
        ego_state[0] = ego_state[0] + ego_state[2] * np.cos(ego_state[3]) * self.dt
        ego_state[1] = ego_state[1] + ego_state[2] * np.sin(ego_state[3]) * self.dt
        ego_state[2] = ego_state[2] + self.control[0] * self.dt
        ego_state[3] = ego_state[3] + self.control[0] * np.tan(self.control[1]) / L * self.dt
        x_mpc, u_mpc = mpc.step(ego_state, target, self.control, verbose=False)
        self.control[0] = u_mpc.value[0, 0]
        self.control[1] = u_mpc.value[1, 0]
        for i in range(self.model_predicted_num):
            self.model_prediction_x.append(x_mpc.value[0,i]*math.cos(self.target_yaw)-x_mpc.value[1,i]*math.sin(self.target_yaw)+self.target_x)
            self.model_prediction_y.append(x_mpc.value[1,i]*math.cos(self.target_yaw)+x_mpc.value[0,i]*math.sin(self.target_yaw)+self.target_y)
        print(self.model_prediction_x)
        print(self.model_prediction_y)


    def detected_object_callback(self,data):
        self.Prediction_policy='heuristic'
        rospy.loginfo("DetectedObjectArray received")
        self.target_veh_dic_x={}
        self.target_veh_dic_y={}
        self.obj=[]
        self.objects_data=data
        for obj in data.objects:
            self.target_velocity_x=obj.velocity.linear.x
            self.target_velocity_y=obj.velocity.linear.y
            self.target_x = obj.pose.position.x
            self.target_y = obj.pose.position.y
            self.target_orientation_x=obj.pose.orientation.x 
            self.target_orientation_y=obj.pose.orientation.y 
            self.target_orientation_z=obj.pose.orientation.z 
            self.target_orientation_w=obj.pose.orientation.w
            self.target_yaw = obj.velocity.angular.z
            self.target_velocity=(self.target_velocity_x**2+self.target_velocity_y**2)**(1/2)
            self.target_yaw=math.atan2(self.target_velocity_y,self.target_velocity_x )
            if self.Prediction_policy=='heuristic':
                self.detected_object_heuristic(obj.label)
            if self.Prediction_policy=='MPC':
                self.prediction_to_object_MPC(obj.label)
            self.obj.append(obj.label)
            print(self.target_veh_dic_x)
            print(self.target_veh_dic_y)
            self.prediction_to_object(obj)

    def detected_object_heuristic(self,obj):
        model_prediction_x = self.target_x
        target_velocity_x = self.target_velocity_x
        target_yaw = self.target_yaw
        model_prediction_y = self.target_y
        target_velocity_y = self.target_velocity_y
        target_angular_z = self.target_angular_z
        self.model_prediction_x = []
        self.model_prediction_y = []
        for i in range(self.model_predicted_num):
            self.model_prediction_x.append(model_prediction_x)
            model_prediction_x += target_velocity_x
            target_yaw += target_angular_z
            target_velocity_x = self.target_velocity * math.cos(target_yaw)
        self.target_veh_dic_x[obj]=self.model_prediction_x
        for j in range(self.model_predicted_num):
            self.model_prediction_y.append(model_prediction_y)
            model_prediction_y += target_velocity_y
            target_yaw += target_angular_z
            target_velocity_y = self.target_velocity * math.sin(target_yaw)
        self.target_veh_dic_y[obj]=self.model_prediction_y

    def prediction_to_object_MPC(self,obj):
        T=1 #Prediction Horizon in Time
        L=0.3 #Vehicle_Wheelbase
        Q = [20, 20, 10, 20]  # state error cost [x,y,v,yaw]
        Qf = [30, 30, 30, 30]  # state error cost at final timestep [x,y,v,yaw]
        R = [10, 10]  # input cost [acc ,steer]
        P = [10, 10]
        start_time=time.time()  # input rate of change cost [acc ,steer]
        mpc = MPC(VehicleModel(), T, self.dt, Q, Qf, R, P)
        if len(self.model_prediction_x)>=10:
            self.model_prediction_x = []
            self.model_prediction_y = []
        self.state[0]=self.target_x
        self.state[1]=self.target_y
        self.state[2]=self.target_velocity
        self.state[3]=self.target_yaw
        target = utils.get_ref_trajectory(self.state, self.path, self.target_velocity, T, self.dt)
        ego_state = np.array([0.0, 0.0, self.state[2], 0.0])
        ego_state[0] = ego_state[0] + ego_state[2] * np.cos(ego_state[3]) * self.dt
        ego_state[1] = ego_state[1] + ego_state[2] * np.sin(ego_state[3]) * self.dt
        ego_state[2] = ego_state[2] + self.control[0] * self.dt
        ego_state[3] = ego_state[3] + self.control[0] * np.tan(self.control[1]) / L * self.dt
        x_mpc, u_mpc = mpc.step(ego_state, target, self.control, verbose=False)
        self.control[0] = u_mpc.value[0, 0]
        self.control[1] = u_mpc.value[1, 0]
        for i in range(self.model_predicted_num):
            self.model_prediction_x.append(x_mpc.value[0,i]*math.cos(self.target_yaw)-x_mpc.value[1,i]*math.sin(self.target_yaw)+self.target_x)
            self.model_prediction_y.append(x_mpc.value[1,i]*math.cos(self.target_yaw)+x_mpc.value[0,i]*math.sin(self.target_yaw)+self.target_y)
        self.target_veh_dic_x[obj]=self.model_prediction_x
        self.target_veh_dic_y[obj]=self.model_prediction_y
        print(self.model_prediction_x)
        print(self.model_prediction_y)

    def prediction_to_object(self,obj):
        Object=DetectedObjectArray()
        
        for obj in self.objects_data.objects:
            detected_obj = DetectedObject()
            veh_pred_list_x = self.target_veh_dic_x[obj.label]
            veh_pred_list_y = self.target_veh_dic_y[obj.label]
            detected_obj.pose.position.x = veh_pred_list_x[2]
            detected_obj.pose.position.y = veh_pred_list_y[2]
            detected_obj.pose.orientation.x=self.target_orientation_x
            detected_obj.pose.orientation.y=self.target_orientation_y
            detected_obj.pose.orientation.z=self.target_orientation_z
            detected_obj.pose.orientation.w=self.target_orientation_w
            detected_obj.dimensions.x=obj.dimensions.x+abs(veh_pred_list_x[4]-veh_pred_list_x[0])*self.bubble_param
            detected_obj.dimensions.y=obj.dimensions.y+abs(veh_pred_list_y[4]-veh_pred_list_y[0])*self.bubble_param
            self.first_point=[veh_pred_list_x[2]+math.cos(self.target_yaw)*(detected_obj.dimensions.x)/2-math.sin(self.target_yaw)*(detected_obj.dimensions.y)/2,veh_pred_list_y[2]+math.sin(self.target_yaw)*(detected_obj.dimensions.x)/2+math.cos(self.target_yaw)*(detected_obj.dimensions.y)/2]
            self.second_point=[veh_pred_list_x[2]+math.cos(self.target_yaw)*(-detected_obj.dimensions.x)/2-math.sin(self.target_yaw)*(detected_obj.dimensions.y)/2,veh_pred_list_y[2]+math.sin(self.target_yaw)*(-detected_obj.dimensions.x)/2+math.cos(self.target_yaw)*(detected_obj.dimensions.y)/2]
            self.third_point=[veh_pred_list_x[2]+math.cos(self.target_yaw)*(-detected_obj.dimensions.x)/2-math.sin(self.target_yaw)*(-detected_obj.dimensions.y)/2,veh_pred_list_y[2]+math.sin(self.target_yaw)*(-detected_obj.dimensions.x)/2+math.cos(self.target_yaw)*(-detected_obj.dimensions.y)/2]
            self.fourth_point=[veh_pred_list_x[2]+math.cos(self.target_yaw)*(detected_obj.dimensions.x)/2-math.sin(self.target_yaw)*(detected_obj.dimensions.y)/2,veh_pred_list_y[2]+math.sin(self.target_yaw)*(detected_obj.dimensions.x)/2+math.cos(self.target_yaw)*(detected_obj.dimensions.y)/2]
            self.Points_x=[]
            self.Points_y=[]
            self.Points_x.append(veh_pred_list_x[2]+math.cos(self.target_yaw)*(detected_obj.dimensions.x)/2-math.sin(self.target_yaw)*(detected_obj.dimensions.y)/2)
            self.Points_x.append(veh_pred_list_x[2]+math.cos(self.target_yaw)*(-detected_obj.dimensions.x)/2-math.sin(self.target_yaw)*(detected_obj.dimensions.y)/2)
            self.Points_x.append(veh_pred_list_x[2]+math.cos(self.target_yaw)*(-detected_obj.dimensions.x)/2-math.sin(self.target_yaw)*(-detected_obj.dimensions.y)/2)
            self.Points_x.append(veh_pred_list_x[2]+math.cos(self.target_yaw)*(detected_obj.dimensions.x)/2-math.sin(self.target_yaw)*(-detected_obj.dimensions.y)/2)
            self.Points_y.append(veh_pred_list_y[2]+math.sin(self.target_yaw)*(detected_obj.dimensions.x)/2+math.cos(self.target_yaw)*(detected_obj.dimensions.y)/2)
            self.Points_y.append(veh_pred_list_y[2]+math.sin(self.target_yaw)*(-detected_obj.dimensions.x)/2+math.cos(self.target_yaw)*(detected_obj.dimensions.y)/2)
            self.Points_y.append(veh_pred_list_y[2]+math.sin(self.target_yaw)*(-detected_obj.dimensions.x)/2+math.cos(self.target_yaw)*(-detected_obj.dimensions.y)/2)
            self.Points_y.append(veh_pred_list_y[2]+math.sin(self.target_yaw)*(detected_obj.dimensions.x)/2+math.cos(self.target_yaw)*(-detected_obj.dimensions.y)/2)
            self.Points_x.append(veh_pred_list_x[2]+math.cos(self.target_yaw)*(detected_obj.dimensions.x)/2-math.sin(self.target_yaw)*(detected_obj.dimensions.y)/2)
            self.Points_y.append(veh_pred_list_y[2]+math.sin(self.target_yaw)*(detected_obj.dimensions.x)/2+math.cos(self.target_yaw)*(detected_obj.dimensions.y)/2)
            Object.objects.append(detected_obj)
        
        self.target_object_pub.publish(Object)

    # def Artificial_potential_field(self):
    #     self.radius=10
    #     for obj in self.objects_data.objects:
    #         potential_field_point=[]
    #         veh_pred_list_x = self.target_veh_dic_x[obj.label]
    #         veh_pred_list_y = self.target_veh_dic_y[obj.label]
    #         for i in len(self.cx):
    #             if abs(veh_pred_list_x[2]+veh_pred_list_y[2]-self.cx[i]-self.cy[i])< 10:
    #                 potential_field_point.append([self.cx[i],self.cy[i]])
    #         print(potential_field_point)

    # def gaussian(self,x,mean,sigma):
    #     return(1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mean)**2/(2*sigma**2)))


    # def simulation_Artificial_potential_field_gaussian(self):
    #     self.radius=10
    #     potential_field_point=[]
    #     repulsed_potential_field_point=[]
    #     self.repulsed_potential_field_point_x=[]
    #     self.repulsed_potential_field_point_y=[]
    #     for i in range(len(self.cx)):
    #         if ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5< self.radius:
    #             potential_field_point.append([self.cx[i],self.cy[i]])
    #     print(len(potential_field_point))
    #     for j in potential_field_point:
    #         x_repulsion=0
    #         y_repulsion=0
    #         for i in range(self.model_predicted_num):
    #             x_repulsion+=self.gaussian(j[0],self.model_prediction_x[i],i+1)
    #             y_repulsion+=self.gaussian(j[1],self.model_prediction_y[i],i+1)
    #         self.repulsed_x=x_repulsion+j[0]
    #         self.repulsed_y=y_repulsion+j[1]
    #         self.repulsed_potential_field_point_x.append(self.repulsed_x)
    #         self.repulsed_potential_field_point_y.append(self.repulsed_y)
    #         repulsed_potential_field_point.append([self.repulsed_x,self.repulsed_y])



            
    # def simulation_Artificial_potential_field_1(self):
    #     self.radius=10
    #     potential_field_point=[]
    #     repulsed_potential_field_point=[]
    #     self.repulsed_potential_field_point_x=[]
    #     self.repulsed_potential_field_point_y=[]
    #     for i in range(len(self.cx)):
    #         if ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5< self.radius:
    #             potential_field_point.append([self.cx[i],self.cy[i]])
    #     print(len(potential_field_point))
    #     for j in potential_field_point:
    #         x_repulsion=0
    #         y_repulsion=0
    #         for i in range(self.model_predicted_num):
    #             if j[0]-self.model_prediction_x[i]<0:
    #                 x_repulsion-=self.gaussian(j[0],self.model_prediction_x[i],i+1)
    #                 y_repulsion-=self.gaussian(j[1],self.model_prediction_y[i],i+1)
    #             if j[0]-self.model_prediction_x[i]>0:
    #                 x_repulsion-=self.gaussian(j[0],self.model_prediction_x[i],i+1)    
    #                 y_repulsion+=self.gaussian(j[1],self.model_prediction_y[i],i+1)
    #         self.repulsed_x=x_repulsion+j[0]
    #         self.repulsed_y=y_repulsion+j[1]
    #         self.repulsed_potential_field_point_x.append(self.repulsed_x)
    #         self.repulsed_potential_field_point_y.append(self.repulsed_y)
    #         repulsed_potential_field_point.append([self.repulsed_x,self.repulsed_y])


    # def repulsive_Artificial_potential_field(self):
    #     self.radius=10
    #     self.gain=1
    #     potential_field_point=[]
    #     repulsed_potential_field_point=[]
    #     self.repulsed_potential_field_point_x=[]
    #     self.repulsed_potential_field_point_y=[]
    #     for i in range(len(self.cx)):
    #         if ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5< self.radius:
    #             potential_field_point.append([self.cx[i],self.cy[i]])
    #     print(len(potential_field_point))
    #     for j in potential_field_point:
    #         x_repulsion=0
    #         y_repulsion=0
    #         for i in range(self.model_predicted_num):
    #             distance=((j[1]-self.model_prediction_y[i])**2+(j[0]-self.model_prediction_x[i])**2)**(1/2)
    #             x_repulsion += self.gain * (1 / distance - 1 / self.radius) * distance**(-3/2) * (j[0] - self.model_prediction_x[i])*(1/i)
    #             y_repulsion += self.gain * (1 / distance - 1 / self.radius) * distance**(-3/2) * (j[1] - self.model_prediction_y[i])*(1/i)
    #         self.repulsed_x=x_repulsion+j[0]
    #         self.repulsed_y=y_repulsion+j[1]
    #         self.repulsed_potential_field_point_x.append(self.repulsed_x)
    #         self.repulsed_potential_field_point_y.append(self.repulsed_y)
    #         repulsed_potential_field_point.append([self.repulsed_x,self.repulsed_y])

    # def Intuitive_Artifical_potential_field(self):
    #     self.radius=10
    #     self.gain=1
    #     potential_field_point=[]
    #     repulsed_potential_field_point=[]
    #     self.repulsed_potential_field_point_x=[]
    #     self.repulsed_potential_field_point_y=[]
    #     for i in range(len(self.cx)):
    #         if ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5< self.radius:
    #             potential_field_point.append([self.cx[i],self.cy[i]])
    #     print(len(potential_field_point))
    #     for j in potential_field_point:
    #         x_repulsion=0
    #         y_repulsion=0
    #         for i in range(self.model_predicted_num):
    #             distance=((j[1]-self.model_prediction_y[i])**2+(j[0]-self.model_prediction_x[i])**2)**(1/2)
    #             x_repulsion += (self.radius-self.gain * ((j[1]-self.model_prediction_y[i])**2))**(1/2)
    #             y_repulsion += (self.radius-self.gain * ((j[0]-self.model_prediction_x[i])**2))**(1/2)
    #         self.repulsed_x=x_repulsion+j[0]
    #         self.repulsed_y=y_repulsion+j[1]
    #         self.repulsed_potential_field_point_x.append(self.repulsed_x)
    #         self.repulsed_potential_field_point_y.append(self.repulsed_y)
    #         repulsed_potential_field_point.append([self.repulsed_x,self.repulsed_y])

    def cartesian_to_frenet(self, centerline, point):
    # 중심선의 아크 길이 계산
        centerline = np.array(centerline)
        diffs = np.diff(centerline, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        arclength = np.insert(np.cumsum(dists), 0, 0.0)

        # 점에서 각 선분까지의 거리 계산 및 최소 거리 찾기
        point = np.array(point)
        min_dist = float('inf')
        s, l = 0, 0

        for i in range(len(diffs)):
            p1 = centerline[i]
            p2 = centerline[i + 1]

            # 점과 선분 사이의 수직 거리 계산
            line_vec = p2 - p1
            point_vec = point - p1
            line_len = np.linalg.norm(line_vec)
            proj_length = np.dot(point_vec, line_vec) / line_len
            proj_point = p1 + (proj_length / line_len) * line_vec
            
            dist = np.linalg.norm(point - proj_point)
            if dist < min_dist:
                min_dist = dist
                s = arclength[i] + proj_length
                l = dist
        
        return s, l
      
    def frenet_to_cartesian(self, centerline, s, l):
    # 중심선의 아크 길이 계산
        centerline = np.array(centerline)
        diffs = np.diff(centerline, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        arclength = np.insert(np.cumsum(dists), 0, 0.0)

        # s에 해당하는 세그먼트 인덱스 찾기
        segment_index = np.searchsorted(arclength, s) - 1
        if segment_index < 0:
            segment_index = 0
        elif segment_index >= len(centerline) - 1:
            segment_index = len(centerline) - 2

        p1 = centerline[segment_index]
        p2 = centerline[segment_index + 1]

        # 세그먼트의 방향 벡터 및 단위 벡터 계산
        segment_vector = p2 - p1
        segment_length = dists[segment_index]
        segment_unit_vector = segment_vector / segment_length

        # s에 대한 기본점 계산
        base_point = p1 + segment_unit_vector * (s - arclength[segment_index])

        # 법선 벡터 계산 (세그먼트에 수직)
        normal_vector = np.array([-segment_unit_vector[1], segment_unit_vector[0]])

        # 카르테시안 좌표 계산
        cartesian_point = base_point + normal_vector * l

        return cartesian_point[0], cartesian_point[1]

    # def Intuitive_Artifical_potential_field_1(self): #Gaussian
    #     self.radius=10
    #     self.gain=1
    #     self.sigma=1
    #     potential_field_point=[]
    #     range_point_x=[]
    #     range_point_y=[]
    #     repulsed_potential_field_point=[]
    #     self.repulsed_potential_field_point_x=[]
    #     self.repulsed_potential_field_point_y=[]
    #     for i in range(len(self.cx)):
    #         if ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5< self.radius:
    #             potential_field_point.append([self.cx[i],self.cy[i]])
    #         if ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5< self.radius*2 and ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5> self.radius:
    #             range_point_x.append(self.cx[i])
    #             range_point_y.append(self.cy[i])
    #     print(len(potential_field_point))
    #     point=[self.model_prediction_x[(self.model_predicted_num)//2],self.model_prediction_y[(self.model_predicted_num)//2]]
    #     center_s,center_d=self.cartesian_to_frenet(potential_field_point,point)
    #     for j in potential_field_point:
    #         s_repulsion=0
    #         d_repulsion=0
    #         for i in range(self.model_predicted_num):
    #             s,d=self.cartesian_to_frenet(potential_field_point,j)
    #             d+=self.gain*self.gaussian(s,center_s,self.sigma)
    #         self.repulsed_s=s_repulsion+s
    #         self.repulsed_d=d_repulsion+d
    #         self.repulsed_x,self.repulsed_y=self.frenet_to_cartesian(potential_field_point,self.repulsed_s,self.repulsed_d)
    #         self.repulsed_potential_field_point_x.append(self.repulsed_x)
    #         self.repulsed_potential_field_point_y.append(self.repulsed_y)


 
    def Intuitive_Artifical_potential_field_2(self):
        # self.local_cx=[]
        # self.local_cy=[]
        # for i in self.cx:    
        #     self.local_cx.append(i)
        # for i in self.cy:
        #     self.local_cy.append(i)# Oval
        # self.radius=10
        start_time=time.time()
        self.gain=0.5
        self.sigma=0.01
        self.radius=round(np.sqrt(1/self.sigma)+1,3)
        print(self.radius)
        potential_field_point=[]
        self.range_point_x=[]
        self.range_point_y=[]
        self.repulsed_potential_field_point_x=[]
        self.repulsed_potential_field_point_y=[]
        for i in range(len(self.cx)):
            if ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5< self.radius:
                potential_field_point.append([self.cx[i],self.cy[i]])
            if ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5< self.radius*2 and ((self.model_prediction_x[2]-self.cx[i])**2+(self.model_prediction_y[2]-self.cy[i])**2)**0.5> self.radius:
                self.range_point_x.append(self.cx[i])
                self.range_point_y.append(self.cy[i])
                
        print(len(potential_field_point))
        point=[self.model_prediction_x[(self.model_predicted_num)//2],self.model_prediction_y[(self.model_predicted_num)//2]]
        center_s,center_d=self.cartesian_to_frenet(potential_field_point,point)
        s1,d1=self.cartesian_to_frenet(potential_field_point,[self.model_prediction_x[2],self.model_prediction_y[2]])
        for j in potential_field_point:
            s,d=self.cartesian_to_frenet(potential_field_point,j)
            for i in range(self.model_predicted_num):
                if d1>=0:
                    d-=self.gain*np.sqrt(np.maximum(0,(1-self.sigma*(s-center_s)**2)))
                else:
                    d+=self.gain*np.sqrt(np.maximum(0,(1-self.sigma*(s-center_s)**2)))
            self.repulsed_s=s
            self.repulsed_d=d
            self.repulsed_x,self.repulsed_y=self.frenet_to_cartesian(potential_field_point,self.repulsed_s,self.repulsed_d)
            self.repulsed_potential_field_point_x.append(self.repulsed_x)
            self.repulsed_potential_field_point_y.append(self.repulsed_y)
        self.Bezier_Curve()
        end_time=time.time()
        print("Execution time:", end_time - start_time)
    # def Intuitive_Artifical_potential_field_2(self):
        # self.gain = 0.5
        # self.sigma = 0.01
        # self.radius = np.sqrt(1 / self.sigma) + 1

        # # 초기화
        # potential_field_point = []
        # self.range_point_x = []
        # self.range_point_y = []
        # self.repulsed_potential_field_point_x = []
        # self.repulsed_potential_field_point_y = []

        # # 잠재적 장벽 필드 포인트 계산
        # for i in range(len(self.cx)):
        #     dist_sq = (self.model_prediction_x[2] - self.cx[i])**2 + (self.model_prediction_y[2] - self.cy[i])**2
        #     if dist_sq < self.radius**2:
        #         potential_field_point.append([self.cx[i], self.cy[i]])
        #     elif self.radius**2 <= dist_sq < (self.radius*2)**2:
        #         self.range_point_x.append(self.cx[i])
        #         self.range_point_y.append(self.cy[i])

        # print(len(potential_field_point))

        # # 중앙점 계산
        # point = [self.model_prediction_x[self.model_predicted_num // 2], self.model_prediction_y[self.model_predicted_num // 2]]
        # center_s, center_d = self.cartesian_to_frenet(potential_field_point, point)
        # s1, d1 = self.cartesian_to_frenet(potential_field_point, [self.model_prediction_x[2], self.model_prediction_y[2]])

        # # 잠재적 장벽 필드 포인트에 대한 반발 계산
        # for j in potential_field_point:
        #     s, d = self.cartesian_to_frenet(potential_field_point, j)
        #     if d1 >= 0:
        #         d -= self.gain * np.sqrt(np.maximum(0, 1 - self.sigma * (s - center_s)**2))
        #     else:
        #         d += self.gain * np.sqrt(np.maximum(0, 1 - self.sigma * (s - center_s)**2))
            
        #     self.repulsed_s = s
        #     self.repulsed_d = d
        #     self.repulsed_x, self.repulsed_y = self.frenet_to_cartesian(potential_field_point, self.repulsed_s, self.repulsed_d)
        #     self.repulsed_potential_field_point_x.append(self.repulsed_x)
        #     self.repulsed_potential_field_point_y.append(self.repulsed_y)
        
        # self.Bezier_Curve()

            
    
    def Bezier_Curve(self):
    # Bezier 처리
        repulsed_x = np.array(self.repulsed_potential_field_point_x)
        repulsed_y = np.array(self.repulsed_potential_field_point_y)
        
        num = len(repulsed_x)
        print(num)
        
        # 점 계산 (각각 1/3, 2/3 위치에서)
        indices = [(num-1) // 3, (num-1) * 2 // 3]
        self.first_point = [repulsed_x[0], repulsed_y[0]]
        self.second_point = [repulsed_x[indices[0]], repulsed_y[indices[0]]]
        self.third_point = [repulsed_x[indices[1]], repulsed_y[indices[1]]]
        self.fourth_point = [repulsed_x[-1], repulsed_y[-1]]
        
        print(self.first_point, self.second_point, self.third_point, self.fourth_point)
        
        # Bezier 곡선 계산
        self.B = self.calc_curve(10)
        
        # 범위 포인트 추가
        self.B[0].extend(self.range_point_x)
        self.B[1].extend(self.range_point_y)
        
        # Trajectory 메시지 생성 및 퍼블리시
        trajectory = LaneArray()
        trajectory.lanes.extend(self.lane_data)
        
        for lane in trajectory.lanes:
            for x, y in zip(self.B[0], self.B[1]):
                waypoint = Waypoint()
                waypoint.pose.pose.position.x = x
                waypoint.pose.pose.position.y = y
                lane.waypoints.append(waypoint)
        
        self.overtaking_traj_pub.publish(trajectory)
        
        print("X coordinates of the Bezier curve:", self.B[0])
        print("Y coordinates of the Bezier curve:", self.B[1])
        
    def calc_curve(self, granularity=10):
        'Calculate the cubic Bezier curve with the given granularity.'
        t_values = np.linspace(0, 1, granularity)
        
        # Precompute the coefficients for the cubic Bezier curve
        coeff1 = (1 - t_values) ** 3
        coeff2 = 3 * (1 - t_values) ** 2 * t_values
        coeff3 = 3 * (1 - t_values) * (t_values ** 2)
        coeff4 = t_values ** 3
        
        # Calculate x and y coordinates using vectorized operations
        B_x = (coeff1 * self.first_point[0] + 
            coeff2 * self.second_point[0] + 
            coeff3 * self.third_point[0] + 
            coeff4 * self.fourth_point[0])
        
        B_y = (coeff1 * self.first_point[1] + 
            coeff2 * self.second_point[1] + 
            coeff3 * self.third_point[1] + 
            coeff4 * self.fourth_point[1])

        return [B_x.tolist(), B_y.tolist()]
    

    
    
             
        
            

if __name__ == '__main__':
    try:
        Model_Predictive()
    except rospy.ROSInterruptException:
        pass