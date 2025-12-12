[backstepping_controller.py](https://github.com/user-attachments/files/24137760/backstepping_controller.py)
#!/usr/bin/env python3
"""
Backstepping 6-DOF controller para BlueROV2 (corregido para ROS2 Jazzy)

- Suscribe odometría (ajusta `odom_topic` si tu simulador publica en otro topic)
- Implementa controlador backstepping 6DOF, mapea tau -> thrusters con B_pinv
- Logging CSV y plot al detener con Ctrl+C
- Corregido: no usa info_once() (incompatible con rclpy en Jazzy)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import time

def quat_to_euler(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    # roll (phi)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    # pitch (theta)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = math.asin(t2)
    # yaw (psi)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

class BacksteppingController(Node):

    def __init__(self):
        super().__init__("backstepping_controller_backstepping")

        # Ajusta si tu simulador publica en otro topic
        odom_topic = "/bluerov2/odom"
        thruster_base = "/bluerov2/cmd_thruster"

        # Publishers thrusters (1..6)
        self.thruster_pubs = [
            self.create_publisher(Float64, f"{thruster_base}{i}", 10)
            for i in range(1, 7)
        ]

        # Subscription ODOMETRY
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_cb, 10
        )

        self.get_logger().info(f"✔ Suscrito a: {odom_topic}")
        self.get_logger().info(f"✔ Publicando a: {thruster_base}X")

        # =========================
        # Estados
        # =========================
        self.eta = np.zeros(6)   # [x,y,z,phi,theta,psi]
        self.nu = np.zeros(6)    # [u,v,w,p,q,r]
        self.odom_received = False

        # =========================
        # Referencia (puedes exponer como ROS param)
        # =========================
        self.eta_r = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        self.eta_r_dot = np.zeros(6)
        self.eta_r_ddot = np.zeros(6)

        # =========================
        # Parámetros vehículo
        # =========================
        self.m = 25.0
        self.Ix = 0.26
        self.Iy = 0.23
        self.Iz = 0.37

        # added mass (aprox)
        self.Xu_dot = 5.5
        self.Yv_dot = 12.7
        self.Zw_dot = 14.57
        self.Kp_dot = 0.12
        self.Mq_dot = 0.12
        self.Nr_dot = 0.12

        # damping (linear approx)
        self.Xu = 25.15
        self.Yv = 7.364
        self.Zw = 17.955
        self.Kp = 10.888
        self.Mq = 20.761
        self.Nr = 3.744

        # =========================
        # Ganancias Backstepping
        # =========================
        self.Lambda = np.diag([10.0, 10.0, 1.0, 10.0, 1.0, 15.0])   # ajustar según comportamiento
        self.Kd = np.diag([15.0, 15.0, 4.0, 8.0, 4.0, 10.0])       # ajustar según comportamiento

        # límtes thrusters (individuales)
        self.tmin = np.array([-40, -40, -40, -40, -40, -40], float)
        self.tmax = np.array([ 40,  40,  40,  40,  40,  40], float)

        # allocation matrix (45° thrusters horizontales, verticales 5 y 6)
        L = 0.18
        c = np.cos(np.deg2rad(45))
        s = np.sin(np.deg2rad(45))
        self.B = np.array([
            [ c,  c,  c,  c,   0,  0],   # Fx
            [ s, -s, -s,  s,   0,  0],   # Fy
            [ 0,  0,  0,  0,  1,  -1],   # Fz (según thrust_dir en SDF)
            [ 0,  0,  0,  0,   0,  0],   # Mx
            [ 0,  0,  0,  0,   0,  0],   # My
            [ L, -L,  L, -L,   0,  0],   # Mz
        ], float)
        self.B_pinv = np.linalg.pinv(self.B)

        # CSV logging
        self.csv_file = open("backstepping_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["t", "x", "y", "z", "phi", "theta", "psi",
                                   "xd", "yd", "zd", "phid", "thetad", "psid",
                                   "Fx", "Fy", "Fz", "Mx", "My", "Mz",
                                   "u0","u1","u2","u3","u4","u5"])
        self.csv_file.flush()

        # tiempo y timer
        self.dt = 0.02
        self.t = 0.0
        self.log_counter = 0
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Backstepping 6-DOF controller started")

    # -------------------------
    # ODOM callback
    # -------------------------
    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        phi, theta, psi = quat_to_euler(msg.pose.pose.orientation)
        self.eta = np.array([x, y, z, phi, theta, psi])

        u = msg.twist.twist.linear.x
        v = msg.twist.twist.linear.y
        w = msg.twist.twist.linear.z
        p = msg.twist.twist.angular.x
        q = msg.twist.twist.angular.y
        r = msg.twist.twist.angular.z
        self.nu = np.array([u, v, w, p, q, r])

        self.odom_received = True

    # -------------------------
    # Dinámica: M, C, D, g
    # -------------------------
    def inertia_matrix(self):
        MRB = np.zeros((6,6))
        MRB[0:3,0:3] = self.m * np.eye(3)
        MRB[3:6,3:6] = np.diag([self.Ix, self.Iy, self.Iz])
        MA = np.diag([self.Xu_dot, self.Yv_dot, self.Zw_dot,
                      self.Kp_dot, self.Mq_dot, self.Nr_dot])
        return MRB + MA

    def coriolis_matrix(self, nu):
        u,v,w,p,q,r = nu
        CRB = np.zeros((6,6))
        CRB[0:3,3:6] = np.array([
            [0, self.m*w, -self.m*v],
            [-self.m*w, 0, self.m*u],
            [self.m*v, -self.m*u, 0]
        ])
        CRB[3:6,0:3] = -CRB[0:3,3:6].T

        CA = np.zeros((6,6))
        CA[0,4] = -self.Zw_dot * w
        CA[0,5] =  self.Yv_dot * v
        CA[1,3] =  self.Zw_dot * w
        CA[1,5] = -self.Xu_dot * u
        CA[2,3] = -self.Yv_dot * v
        CA[2,4] =  self.Xu_dot * u

        return CRB + CA

    def damping_matrix(self, nu):
        u,v,w,p,q,r = nu
        D_lin = np.diag([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr])
        return D_lin

    def gravity_vector(self, eta):
        phi, theta = eta[3], eta[4]
        W = self.m * 9.81
        B = self.m * 9.81 * 1.02  # ligera flotabilidad positiva por defecto
        z_B = 0.06
        cphi = math.cos(phi); sphi = math.sin(phi)
        cth = math.cos(theta); sth = math.sin(theta)
        g = np.zeros(6)
        g[2] = (W - B) * cphi * cth
        g[3] = - z_B * B * sth
        g[4] =  z_B * B * sphi * cth
        g[5] = 0.0
        return g

    # J, Jinv y Jdot
    def J_matrix(self, eta):
        phi, theta, psi = eta[3], eta[4], eta[5]
        cphi = math.cos(phi); sphi = math.sin(phi)
        cth = math.cos(theta); sth = math.sin(theta)
        cps = math.cos(psi); sps = math.sin(psi)
        R = np.array([
            [cps*cth, cps*sth*sphi - sps*cphi, cps*sth*cphi + sps*sphi],
            [sps*cth, sps*sth*sphi + cps*cphi, sps*sth*cphi - cps*sphi],
            [-sth,    cth*sphi,                 cth*cphi]
        ])
        cth_safe = max(1e-6, cth)
        J2 = np.array([
            [1.0, sphi * math.tan(theta),  cphi * math.tan(theta)],
            [0.0, cphi,                   -sphi],
            [0.0, sphi/cth_safe,          cphi/cth_safe]
        ])
        J = np.zeros((6,6))
        J[0:3,0:3] = R
        J[3:6,3:6] = J2
        return J

    def J_inv_matrix(self, eta):
        J = self.J_matrix(eta)
        R = J[0:3,0:3]
        J2 = J[3:6,3:6]
        R_inv = R.T
        try:
            J2_inv = np.linalg.inv(J2)
        except np.linalg.LinAlgError:
            J2_inv = np.linalg.pinv(J2)
        Jinv = np.zeros((6,6))
        Jinv[0:3,0:3] = R_inv
        Jinv[3:6,3:6] = J2_inv
        return Jinv

    def J_dot_matrix(self, eta, nu):
        # aproximación numérica robusta (finite differences sobre ángulos)
        eps = 1e-6
        J = self.J_matrix(eta)
        J2 = J[3:6,3:6]
        euler_rates = J2 @ nu[3:6]
        eta_dot_ang = euler_rates
        Jdot = np.zeros_like(J)
        for i in range(3):
            eta_pert = eta.copy()
            eta_pert[3 + i] += eps
            J_pert = self.J_matrix(eta_pert)
            Jdot += (J_pert - J) * (eta_dot_ang[i] / eps)
        return Jdot

    def J_inv_dot_matrix(self, eta, nu):
        J = self.J_matrix(eta)
        Jdot = self.J_dot_matrix(eta, nu)
        Jinv = self.J_inv_matrix(eta)
        Jinv_dot = - Jinv @ Jdot @ Jinv
        return Jinv_dot

    # -------------------------
    # Control loop (backstepping)
    # -------------------------
    def control_loop(self):
        if not self.odom_received:
            # imprimir una vez mientras esperamos odometría
            if not hasattr(self, "printed_waiting_msg"):
                self.get_logger().info("Esperando odometría...")
                self.printed_waiting_msg = True
            return

        eta = self.eta.copy()
        nu = self.nu.copy()

        # dinámicas
        M = self.inertia_matrix()
        C = self.coriolis_matrix(nu)
        D = self.damping_matrix(nu)
        g = self.gravity_vector(eta)

        J = self.J_matrix(eta)
        Jinv = self.J_inv_matrix(eta)
        Jinv_dot = self.J_inv_dot_matrix(eta, nu)

        # errores
        e1 = eta - self.eta_r
        e1_dot = J @ nu - self.eta_r_dot

        # velocidad virtual
        nu_r = Jinv @ (self.eta_r_dot - self.Lambda @ e1)

        # derivada nu_r
        nu_r_dot = Jinv_dot @ (self.eta_r_dot - self.Lambda @ e1) + \
                   Jinv @ (self.eta_r_ddot - self.Lambda @ e1_dot)

        # error de velocidad
        e2 = nu - nu_r

        # ley backstepping
        tau = M @ (nu_r_dot - J @ e1 - self.Kd @ e2) + C @ nu + D @ nu + g

        # mapeo a thrusters
        u = self.B_pinv @ tau
        u_sat = np.minimum(np.maximum(u, self.tmin), self.tmax)

        # logging CSV
        try:
            row = [round(self.t,4),
                   eta[0], eta[1], eta[2], eta[3], eta[4], eta[5],
                   self.eta_r[0], self.eta_r[1], self.eta_r[2], self.eta_r[3], self.eta_r[4], self.eta_r[5],
                   tau[0], tau[1], tau[2], tau[3], tau[4], tau[5],
                   u_sat[0], u_sat[1], u_sat[2], u_sat[3], u_sat[4], u_sat[5]]
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except Exception as e:
            self.get_logger().warning(f"CSV write error: {e}")

        self.t += self.dt

        # logs limitados a consola
        if self.log_counter % 10 == 0:
            self.get_logger().info(
                "\n"
                f"POS=[{eta[0]:.2f},{eta[1]:.2f},{eta[2]:.2f}, {eta[5]:.2f}] \n"
                f"SP    = [{self.eta_r[0]:.2f}, {self.eta_r[1]:.2f}, {self.eta_r[2]:.2f}, {self.eta_r[5]:.2f}]\n"
                f"ERR_pos=[{e1[0]:.2f},{e1[1]:.2f},{e1[2]:.2f}, {e1[5]:.2f}] \n"
                f"TAU=[{tau[0]:.1f},{tau[1]:.1f},{tau[2]:.1f},{tau[5]:.1f}] \n"
                f"THR=[{u_sat[0]:.1f},{u_sat[1]:.1f},{u_sat[2]:.1f},{u_sat[3]:.1f},{u_sat[4]:.1f},{u_sat[5]:.1f}]"
            )
        self.log_counter += 1

        # publicar thrusters
        for i, pub in enumerate(self.thruster_pubs):
            msg = Float64()
            msg.data = float(u_sat[i])
            pub.publish(msg)

    # -------------------------
    # plotting al detener
    # -------------------------
    def plot_results(self):
        try:
            df = pd.read_csv("backstepping_log.csv")
        except Exception as e:
            self.get_logger().error(f"No se pudo leer backstepping_log.csv: {e}")
            return
        if df.empty:
            self.get_logger().warning("CSV vacío")
            return
        t = df["t"]
        x = df["x"]
        y = df["y"]
        z = df["z"]
        yaw = df["psi"]
        xd = df["xd"]
        yd = df["yd"]
        zd = df["zd"]
        yawd = df["psid"]

        u0 = df["u0"]
        u1 = df["u1"]
        u2 = df["u2"]
        u5 = df["u5"]

        plt.figure(figsize=(10, 10))

        plt.subplot(5,1,1)
        plt.plot(t, x, label="x")
        plt.plot(t, xd, "--", label="x_ref")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,2)
        plt.plot(t, y, label="y")
        plt.plot(t, yd, "--", label="y_ref")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,3)
        plt.plot(t, z, label="z")
        plt.plot(t, zd, "--", label="z_ref")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,4)
        plt.plot(t, yaw, label="yaw")
        plt.plot(t, yawd, "--", label="yaw_ref")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,5)
        plt.plot(t, u0, label="u_x")
        plt.plot(t, u1, label="u_y")
        plt.plot(t, u2, label="u_z")
        plt.plot(t, u5, label="u_yaw")
        plt.grid(True)
        plt.legend()

        plt.suptitle("Respuesta del Controlador Backstepping")
        plt.tight_layout()
        plt.show()

    def wrap(self, a):
        return (a + math.pi) % (2 * math.pi) - math.pi

def main(args=None):
    rclpy.init(args=args)
    node = BacksteppingController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n\n>>> CONTROLADOR DETENIDO — Generando gráficas...\n")
        # cerramos CSV y luego graficamos
        try:
            node.csv_file.close()
        except Exception:
            pass
        node.plot_results()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

