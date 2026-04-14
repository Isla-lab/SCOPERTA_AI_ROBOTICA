#!/usr/bin/env python3
import time
import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from ultralytics import YOLO


class LimoYoloDistance(Node):
    def __init__(self):
        super().__init__("limo_yolo_distance_single_target")
        self.get_logger().info("--- Inizializzazione Nodo YOLO Limo (Sweep Sequenziale ANTI-INERZIA) ---")
        
        # ---------------- PARAMETRI DI VELOCITA ----------------
        self.rgb_topic = "/camera/color/image_raw"
        self.target_class = "sports ball"
        self.score_thresh = 0.2
        self.center_threshold = 0.1
        self.forward_speed = 0.3  # Velocità di crociera
        self.rotation_speed = 1.2 # Mantenere bassa per evitare slittamenti!

        # ---------------- TEMPI DELLO SWEEP ----------------
        # Aumenta t_rot se vuoi che l'angolo di ricerca sia più ampio.
        self.t_rot = 2.5      # Tempo di rotazione (es. da centro a sinistra)
        self.t_stop = 0.5      # Tempo di STOP per azzerare l'inerzia (cruciale!)
        self.t_fwd = 1.5       # Tempo di avanzamento dritto
        
        # ---------------- DIMENSIONI & CAMERA ----------------
        self.target_real_width = {
            "sports ball": 0.2
        }
        self.focal_length_px = 200#600.0
        self.distance_threshold = 1.0

        # ---------------- QoS ----------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ---------------- VARIABILI ----------------
        self.rgb_image = None
        self.lock = threading.Lock()
        self.target_detected = False
        self.target_cx = 0
        self.target_cy = 0
        self.target_width = 0
        self.last_seen_time = time.time()
        self.current_distance = None
        
        self.bridge = CvBridge()
        
        # Variabili per lo sweep
        self.is_searching = False
        self.search_start_time = 0.0

        # ---------------- MODELLO ----------------
        self.get_logger().info("Caricamento modello YOLO in corso...")
        self.model = YOLO("yolov8n.pt")
        self.get_logger().info("Modello caricato con successo!")

        # ---------------- ROS ----------------
        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_pub = self.create_publisher(Image, "/yolo/annotated_image", 10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f"Nodo avviato. Target da inseguire: '{self.target_class}'")

    # ---------------- CALLBACK RGB ----------------
    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.rgb_image = img
        except Exception as e:
            self.get_logger().error(f"Errore conversione: {e}")

    # ---------------- DETECTION ----------------
    def detect(self, rgb):
        results = self.model.predict(rgb, conf=self.score_thresh, verbose=False)

        self.target_detected = False
        self.current_distance = None
        debug_img = rgb.copy()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id].lower()
                
                if cls_name != self.target_class:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                width = x2 - x1
                conf = float(box.conf[0])

                real_width = self.target_real_width.get(self.target_class, 0.1)
                distance = (real_width * self.focal_length_px) / max(width, 1e-5)

                self.target_detected = True
                self.target_cx = cx
                self.target_cy = cy
                self.target_width = width
                self.last_seen_time = time.time()
                self.current_distance = distance

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
                label = f"{cls_name} (Conf: {conf:.2f}) {distance:.2f}m"
                cv2.putText(debug_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break 

        if not self.target_detected:
            cv2.putText(debug_img, "NESSUN OGGETTO RILEVATO", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if self.is_searching:
                cv2.putText(debug_img, "SWEEP IN CORSO...", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        self.publish_debug(debug_img)

    # ---------------- CONTROL LOOP ----------------
    def control_loop(self):
        with self.lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None

        if rgb is None:
            return

        self.detect(rgb)
        now = time.time()
        
        # LOGICA SWEEP
        if not self.target_detected and (now - self.last_seen_time > 0.5):
            if not self.is_searching:
                self.is_searching = True
                self.search_start_time = now
                msg = "Target perso: avvio sequenza rigida Sinistra -> Destra -> Centro -> Avanti"
                self.get_logger().warning(msg)
                self.get_logger().info(f"\n[!] {msg}")
            
            elapsed = now - self.search_start_time
            
            # Calcolo dei trigger temporali progressivi
            s1 = self.t_rot                                # Fine Gira Sinistra
            s2 = s1 + self.t_stop                          # Fine STOP 1
            s3 = s2 + (self.t_rot * 2)                     # Fine Gira Destra (x2 per coprire l'altro lato)
            s4 = s3 + self.t_stop                          # Fine STOP 2
            s5 = s4 + self.t_rot                           # Fine Torna Centro
            s6 = s5 + self.t_stop                          # Fine STOP 3
            s7 = s6 + self.t_fwd                           # Fine Avanzamento
            s8 = s7 + self.t_stop                          # Fine STOP 4
            
            # Macchina a stati esatta
            if elapsed < s1:
                self.publish_twist(0.0, self.rotation_speed)         # 1. Gira SX
            elif elapsed < s2:
                self.publish_stop()                                  # 2. STOP
            elif elapsed < s3:
                self.publish_twist(0.0, -self.rotation_speed)        # 3. Gira DX
            elif elapsed < s4:
                self.publish_stop()                                  # 4. STOP
            elif elapsed < s5:
                self.publish_twist(0.0, self.rotation_speed)         # 5. Torna Centro
            elif elapsed < s6:
                self.publish_stop()                                  # 6. STOP
            elif elapsed < s7:
                self.publish_twist(self.forward_speed, 0.0)          # 7. Avanza
            elif elapsed < s8:
                self.publish_stop()                                  # 8. STOP
            else:
                self.search_start_time = now                         # 9. Ripeti tutto
            return 
        
        # LOGICA INSEGUIMENTO
        if self.target_detected:
            if self.is_searching:
                self.is_searching = False
                msg = "OGGETTO RILEVATO: interrompo sweep e inseguo."
                self.get_logger().info(msg)
                self.get_logger().info(f"\n[OK] {msg}")

            h, w = rgb.shape[:2]
            error = (self.target_cx - w / 2) / (w / 2)

            if abs(error) > self.center_threshold:
                self.publish_twist(0.0, -self.rotation_speed * error)
                return

            if self.current_distance > self.distance_threshold:
                self.get_logger().info(f"Distanza {self.current_distance:.2f}")
                self.publish_twist(self.forward_speed, 0.0)
            else:
                self.publish_stop()
                if int(now * 10) % 10 == 0: 
                    self.get_logger().info(f"[ARRIVATO] Target a distanza: {self.current_distance:.2f}m. Attesa...")

    # ---------------- UTILS ----------------
    def publish_debug(self, img):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            self.debug_pub.publish(msg)
        except Exception as e:
            pass

    def publish_twist(self, lin, ang):
        msg = Twist()
        msg.linear.x = lin
        msg.angular.z = ang
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        self.publish_twist(0.0, 0.0)


def main():
    rclpy.init()
    node = LimoYoloDistance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        self.get_logger().info("\nArresto manuale richiesto.")
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()