import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from tf.transformations import euler_from_quaternion
import numpy as np
from hmcl_msgs.msg import LaneArray
import time
from hmcl_msgs.msg import Waypoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50).to(device)
        c_0 = torch.zeros(2, x.size(0), 50).to(device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out
    
class Model_Predictive():
    def __init__(self):
        rospy.init_node("Model_Predictive")
        self.target_pose_sub = rospy.Subscriber('/target_pose', Pose, self.target_pose_callback)
        self.target_velocity_sub = rospy.Subscriber('/target_velocity', Twist, self.target_velocity_callback)
        self.target_angular_sub = rospy.Subscriber('/target_angular', Float64, self.target_angular_callback)
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
        self.vx_data = []
        self.vy_data = []
        self.target_x_data=[]
        self.target_y_data=[]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.plot_done = False
        self.rate = rospy.Rate(10)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # PyTorch 모델 초기화
        self.model = LSTMModel(input_size=1, hidden_size=50, output_size=1).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
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
            plt.pause(0.1)
    
        
            
            
    def global_traj_callback(self,data):
        self.cx=[]
        self.cy=[]
        for i in data.lanes:
            for j in i.waypoints:
                self.cx.append(j.pose.pose.position.x)
                self.cy.append(j.pose.pose.position.y)
        
        
        self.lane_data=data.lanes

            #set mode simulation or Detect(ROS bag)
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
        self.target_x = data.position.x
        self.target_y = data.position.y
        self.target_x_data.append(self.target_x)
        self.target_y_data.append(self.target_y)

    def target_velocity_callback(self, data):
        self.target_velocity_x = data.linear.x
        self.target_velocity_y = data.linear.y
        self.vx_data.append(self.target_velocity_x)
        self.vy_data.append(self.target_velocity_y)
        self.target_velocity=(self.target_velocity_x**2+self.target_velocity_y **2)**(1/2)
        print(f"Data length: {len(self.vx_data)}")
        if len(self.vx_data) >= 1000:
            if not self.plot_done:
                self.make_lstm()
            self.run_lstm()
            self.plot_done = True
            self.Intuitive_Artifical_potential_field_2()


            
    def make_lstm(self):
        print("Running LSTM model for x and y...")

        # Scale the data
        vx_data_array = np.array(self.vx_data).reshape(-1, 1)
        vy_data_array = np.array(self.vy_data).reshape(-1, 1)
        self.vx_data_scaled = self.scaler.fit_transform(vx_data_array)
        self.vy_data_scaled = self.scaler.fit_transform(vy_data_array)

        # Prepare LSTM input data (sequence length 20)
        X_x, y_x = [], []
        X_y, y_y = [], []

        for i in range(20, len(self.vx_data_scaled)):
            X_x.append(self.vx_data_scaled[i-20:i, 0])
            y_x.append(self.vx_data_scaled[i, 0])
            X_y.append(self.vy_data_scaled[i-20:i, 0])
            y_y.append(self.vy_data_scaled[i, 0])

        X_x, y_x = np.array(X_x), np.array(y_x)
        X_y, y_y = np.array(X_y), np.array(y_y)

        X_x = torch.tensor(X_x, dtype=torch.float32).reshape(X_x.shape[0], X_x.shape[1], 1).to(self.device)
        y_x = torch.tensor(y_x, dtype=torch.float32).reshape(-1, 1).to(self.device)

        X_y = torch.tensor(X_y, dtype=torch.float32).reshape(X_y.shape[0], X_y.shape[1], 1).to(self.device)
        y_y = torch.tensor(y_y, dtype=torch.float32).reshape(-1, 1).to(self.device)

        # Create LSTM models for x and y
        self.model_x = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(self.device)
        self.model_y = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(self.device)

        criterion = nn.MSELoss()
        optimizer_x = optim.Adam(self.model_x.parameters(), lr=0.001)
        optimizer_y = optim.Adam(self.model_y.parameters(), lr=0.001)

        # Train the LSTM models
        self.model_x.train()
        self.model_y.train()

        for epoch in range(20):
            for i in range(0, len(X_x), 64):
                X_batch_x = X_x[i:i+64]
                y_batch_x = y_x[i:i+64]

                optimizer_x.zero_grad()
                outputs_x = self.model_x(X_batch_x)
                loss_x = criterion(outputs_x, y_batch_x)
                loss_x.backward()
                optimizer_x.step()

                X_batch_y = X_y[i:i+64]
                y_batch_y = y_y[i:i+64]

                optimizer_y.zero_grad()
                outputs_y = self.model_y(X_batch_y)
                loss_y = criterion(outputs_y, y_batch_y)
                loss_y.backward()
                optimizer_y.step()

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/100], Loss x: {loss_x.item():.4f}, Loss y: {loss_y.item():.4f}')

    def run_lstm(self):
        self.model_prediction_x=[]
        self.model_prediction_y=[]
        self.model_x.eval()
        self.model_y.eval()
        start_time = time.time()

        # Prepare input for x and y predictions
        inputs_x = np.array(self.vx_data[-20:]).reshape(-1, 1)
        inputs_x = self.scaler.transform(inputs_x)
        inputs_x = torch.tensor(inputs_x, dtype=torch.float32).reshape(1, 20, 1).to(self.device)

        inputs_y = np.array(self.vy_data[-20:]).reshape(-1, 1)
        inputs_y = self.scaler.transform(inputs_y)
        inputs_y = torch.tensor(inputs_y, dtype=torch.float32).reshape(1, 20, 1).to(self.device)

        self.forecast_x = []
        self.forecast_y = []
        
        with torch.no_grad():
            for _ in range(5):
                predicted_value_x = self.model_x(inputs_x)
                self.forecast_x.append(predicted_value_x.item())
                inputs_x = torch.cat((inputs_x[:, 1:, :], predicted_value_x.reshape(1, 1, 1)), dim=1)

                predicted_value_y = self.model_y(inputs_y)
                self.forecast_y.append(predicted_value_y.item())
                inputs_y = torch.cat((inputs_y[:, 1:, :], predicted_value_y.reshape(1, 1, 1)), dim=1)

        self.forecast_x = self.scaler.inverse_transform(np.array(self.forecast_x).reshape(-1, 1)).flatten()
        self.forecast_y = self.scaler.inverse_transform(np.array(self.forecast_y).reshape(-1, 1)).flatten()
        
        self.vx_tot=0
        self.vy_tot=0 
        #self.dt에 신호 주기를 입력해야함!!
        for i in range(len(self.forecast_x)):
            self.vx_tot+=self.forecast_x[i]
            self.vy_tot+=self.forecast_y[i]
            self.model_prediction_x.append(self.vx_tot*self.dt+self.target_x)
            self.model_prediction_y.append(self.vy_tot*self.dt+self.target_y)
        actual = np.array(self.vx_data[-5:])
        self.evaluate_forecast(actual, self.forecast_x)
        end_time = time.time()
        print("Forecast: ", self.forecast_x)
        print("Actual: ", actual)
        print(f"Execution Time: {end_time - start_time} seconds")
        
    def evaluate_forecast(self, actual, forecast):
        mae = mean_absolute_error(actual, forecast)
        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    
    def target_angular_callback(self, data):
        self.target_angular_z = data.data

    def get_yaw_from_orientation(self, x, y, z, w):
        euler = euler_from_quaternion([x, y, z, w])
        return euler[2] 





 
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
        self.sigma=1/self.target_velocity
        self.radius=round(np.sqrt(1/self.sigma)+1,3)
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
        print(potential_field_point,point)
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