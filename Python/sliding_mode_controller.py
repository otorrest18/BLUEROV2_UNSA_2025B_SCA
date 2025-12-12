#!/usr/bin/env python3
"""
Controlador SMC (modo deslizante) para BlueROV2
Adaptado para ROS2 Jazzy + gz-sim.

Incluye:
 - Logging automático en smc_log.csv (t, x,y,z,yaw, refs, Fx,Fy,Fz,Mz, u0..u5)
 - Gráficas automáticas al presionar Ctrl+C
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


def quat_to_yaw(q):
    """Convierte quaternion → yaw."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SlidingModeController(Node):

    def __init__(self):
        super().__init__("sliding_mode_controller")

        odom_topic = "/bluerov2/odom"
        thruster_base = "/bluerov2/cmd_thruster"

        self.thruster_pubs = [
            self.create_publisher(Float64, f"{thruster_base}{i}", 10)
            for i in range(1, 7)
        ]

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_cb, 10
        )

        self.get_logger().info(f"✔ Suscrito a: {odom_topic}")
        self.get_logger().info(f"✔ Publicando a: {thruster_base}X")

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.yaw = 0.0
        self.yaw_rate = 0.0

        # Setpoints (puedes modificarlos dinámicamente)
        self.xd = 5.0
        self.yd = 5.0
        self.zd = 0.0
        self.yawd = 1.0

        # Modelo (m, Izz)
        self.m = 25.0
        self.Izz = 13.0

        # Ganancias PD nominales (componente equivalente)
        self.kp_x = 15.0
        self.kd_x = 11.0

        self.kp_y = 15.0
        self.kd_y = 11.0

        self.kp_z = 0.4
        self.kd_z = 60.0

        self.kp_yaw = 15.0
        self.kd_yaw = 11.0

        # ===== Parámetros SMC (ajustables) =====
        self.lambda_x = 1.5
        self.lambda_y = 1.5
        self.lambda_z = 2.0
        self.lambda_yaw = 2.0

        self.eta_x = 60.0
        self.eta_y = 60.0
        self.eta_z = 200.0
        self.eta_yaw = 10.0

        self.phi_x = 0.2
        self.phi_y = 0.2
        self.phi_z = 0.5
        self.phi_yaw = 1.0

        # límites thrusters
        self.tmin = np.array([-40, -40, -40, -40, -40, -40], float)
        self.tmax = np.array([ 40,  40,  40,  40,  40,  40], float)

        # Matriz de asignación thrusters
        L = 0.18
        c = np.cos(np.deg2rad(45))
        s = np.sin(np.deg2rad(45))

        self.B = np.array([
            [ c,  c,  c,  c, 0, 0],   # Fx
            [ s, -s, -s,  s, 0, 0],   # Fy
            [ 0,  0,  0,  0, -1, 1],  # Fz
            [ 0,  0,  0,  0, 0, 0],   # Mx
            [ 0,  0,  0,  0, 0, 0],   # My
            [ L, -L,  L, -L, 0, 0],   # Mz
        ], float)

        self.B_pinv = np.linalg.pinv(self.B)

        # contador para limitar logs
        self.log_counter = 0

        # ============================
        #     CSV LOG AUTOMÁTICO
        # ============================
        # Abrimos el archivo desde el inicio y escribimos cabecera (coincide con columnas usadas)
        self.csv_file = open("smc_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "t",
            "x", "y", "z", "yaw",
            "xd", "yd", "zd", "yawd",
            "Fx", "Fy", "Fz", "Mz",
            "u0", "u1", "u2", "u3", "u4", "u5"
        ])
        self.csv_file.flush()

        # tiempo simulado / muestreo
        self.t = 0.0
        self.dt = 0.02  # 50 Hz

        self.timer = self.create_timer(self.dt, self.control_loop)  # 50 Hz


    def odom_cb(self, msg: Odometry):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        self.pos[2] = msg.pose.pose.position.z

        self.vel[0] = msg.twist.twist.linear.x
        self.vel[1] = msg.twist.twist.linear.y
        self.vel[2] = msg.twist.twist.linear.z

        q = msg.pose.pose.orientation
        self.yaw = quat_to_yaw(q)
        self.yaw_rate = msg.twist.twist.angular.z

    def wrap(self, a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def sat(self, x, phi):
        """Saturación suave: tanh(x/phi). phi>0"""
        if phi <= 0.0:
            return np.sign(x)  # fallback (no smoothing)
        return np.tanh(x / phi)

    def control_loop(self):

        # Errores de posición
        ex = self.xd - self.pos[0]
        ey = self.yd - self.pos[1]
        ez = self.zd - self.pos[2]

        # Errores en velocidad (setpoint vel = 0)
        evx = -self.vel[0]
        evy = -self.vel[1]
        evz = -self.vel[2]

        eyaw = self.wrap(self.yawd - self.yaw)
        eyaw_rate = -self.yaw_rate

        # Superficies de deslizamiento
        sx = evx + self.lambda_x * ex
        sy = evy + self.lambda_y * ey
        sz = evz + self.lambda_z * ez
        syaw = eyaw_rate + self.lambda_yaw * eyaw

        # Componente equivalente (PD nominal)
        Fx_eq = self.m * (self.kp_x * ex + self.kd_x * evx)
        Fy_eq = self.m * (self.kp_y * ey + self.kd_y * evy)
        Fz_eq = self.m * (self.kp_z * ez + self.kd_z * evz)
        Mz_eq = self.Izz * (self.kp_yaw * eyaw + self.kd_yaw * eyaw_rate)

        # Término robusto SMC (suavizado con tanh)
        Fx_robust = - self.eta_x * self.sat(sx, self.phi_x)
        Fy_robust = - self.eta_y * self.sat(sy, self.phi_y)
        Fz_robust = - self.eta_z * self.sat(sz, self.phi_z)
        Mz_robust = - self.eta_yaw * self.sat(syaw, self.phi_yaw)

        # Ley total
        Fx = Fx_eq + Fx_robust
        Fy = Fy_eq + Fy_robust
        Fz = Fz_eq + Fz_robust
        Mz = Mz_eq + Mz_robust

        tau = np.array([Fx, Fy, Fz, 0.0, 0.0, Mz])
        u = self.B_pinv @ tau
        u = np.minimum(np.maximum(u, self.tmin), self.tmax)

        # =============================
        #   LOG DE POSICIÓN, FUERZAS Y COMANDOS (cada ciclo)
        # =============================
        try:
            self.csv_writer.writerow([
                round(self.t, 4),
                float(self.pos[0]), float(self.pos[1]), float(self.pos[2]), float(self.yaw),
                float(self.xd), float(self.yd), float(self.zd), float(self.yawd),
                float(Fx), float(Fy), float(Fz), float(Mz),
                float(u[0]), float(u[1]), float(u[2]), float(u[3]), float(u[4]), float(u[5])
            ])
            # Forzamos a disco (evita archivo vacío si cortas rápido)
            self.csv_file.flush()
        except Exception as e:
            self.get_logger().warning(f"CSV write error: {e}")

        # incrementamos tiempo nominal
        self.t += self.dt

        # =============================
        #   LOGS por consola (limitados)
        # =============================
        if self.log_counter % 10 == 0:  # cada 10 ciclos (~5 Hz)
            self.get_logger().info(
                "\n"
                f"POS   = [{self.pos[0]:.2f}, {self.pos[1]:.2f}, {self.pos[2]:.2f}, {self.yaw:.2f}]\n"
                f"SP    = [{self.xd:.2f}, {self.yd:.2f}, {self.zd:.2f}, {self.yawd:.2f}]\n"
                f"ERR   = [{ex:.2f}, {ey:.2f}, {ez:.2f}, {eyaw:.2f}]\n"
                f"S     = [{sx:.3f}, {sy:.3f}, {sz:.3f}, {syaw:.3f}]\n"
                f"TAU   = [{Fx:.2f}, {Fy:.2f}, {Fz:.2f}, {Mz:.2f}]\n"
                f"THR   = [{u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f}, {u[3]:.2f}, {u[4]:.2f}, {u[5]:.2f}]"
            )

        self.log_counter += 1

        # Publicar en cada thruster
        for i, pub in enumerate(self.thruster_pubs):
            msg = Float64()
            msg.data = float(u[i])
            pub.publish(msg)

    # =======================================================
    #         GRAFICADOR AUTOMÁTICO AL DETENER CONTROLADOR
    # =======================================================
    def plot_results(self):
        # Usamos pandas para conveniencia
        try:
            df = pd.read_csv("smc_log.csv")
        except Exception as e:
            self.get_logger().error(f"No se pudo leer smc_log.csv: {e}")
            return

        if df.empty:
            self.get_logger().warning("smc_log.csv está vacío — no hay datos para graficar.")
            return

        t = df["t"]
        x = df["x"]
        y = df["y"]
        z = df["z"]
        yaw = df["yaw"]
        xd = df["xd"]
        yd = df["yd"]
        zd = df["zd"]
        yawd = df["yawd"]

        Fx = df["Fx"]
        Fy = df["Fy"]
        Fz = df["Fz"]
        Mz = df["Mz"]

        u0 = df["u0"]
        u1 = df["u1"]
        u2 = df["u2"]
        u3 = df["u3"]
        u4 = df["u4"]
        u5 = df["u5"]

        plt.figure(figsize=(10, 12))

        plt.subplot(5,1,1)
        plt.plot(t, x, label="x")
        plt.plot(t, xd, "--", label="x_ref")
        plt.ylabel("X [m]")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,2)
        plt.plot(t, y, label="y")
        plt.plot(t, yd, "--", label="y_ref")
        plt.ylabel("Y [m]")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,3)
        plt.plot(t, z, label="z")
        plt.plot(t, zd, "--", label="z_ref")
        plt.ylabel("Z [m]")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,4)
        plt.plot(t, yaw, label="yaw")
        plt.plot(t, yawd, "--", label="yaw_ref")
        plt.ylabel("Yaw [rad]")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,5)
        plt.plot(t, u0, label="u0")
        plt.plot(t, u1, label="u1")
        plt.plot(t, u2, label="u2")
        plt.plot(t, u3, label="u3")
        plt.plot(t, u4, label="u4")
        plt.plot(t, u5, label="u5")
        plt.ylabel("Thruster cmds")
        plt.xlabel("Tiempo [s]")
        plt.grid(True)
        plt.legend()

        plt.suptitle("Respuesta del SMC")
        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = SlidingModeController()

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
