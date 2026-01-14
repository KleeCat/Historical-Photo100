import os
import cv2
import torch
import numpy as np
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Try importing metrics libraries
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    psnr = None
    ssim = None
    print("Warning: skimage not installed.")

# --- Global Theme Settings ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class ModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Image Super-Resolution System (ESRGAN)")
        self.geometry("1300x900")  # Increased height for new controls

        # Core variables
        # UPDATE THIS FOLDER PATH ONLY
        self.model_folder = r"C:\Users\ihggk\.cache\realesrgan"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.upsampler = None
        self.model = None
        self.scale_factor = 4  # Default
        self.img_input = None
        self.img_output = None
        self.img_gt = None
        self.is_processing = False
        self.feature_maps = {}

        # Layout configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_ui()

        # Start background model loading (Default load x4)
        self.status_label.configure(text=f"Initializing core components ({self.device})...")
        self.progress_bar.set(0.5)
        threading.Thread(target=self.load_model, daemon=True).start()

    def setup_ui(self):
        # === 1. Sidebar (Left) ===
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        # Logo / Title
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Super Resolution", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Controls
        self.btn_load = ctk.CTkButton(self.sidebar, text="📂 Open Image", command=self.load_input_image, height=40)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)

        self.btn_gt = ctk.CTkButton(self.sidebar, text="🖼️ Load Ground Truth", command=self.load_gt_image,
                                    fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), height=40)
        self.btn_gt.grid(row=2, column=0, padx=20, pady=10)

        self.separator = ctk.CTkProgressBar(self.sidebar, height=2, progress_color="gray")
        self.separator.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.separator.set(1)

        # === New Scale Selector ===
        ctk.CTkLabel(self.sidebar, text="Upscale Factor:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=4,
                                                                                                          column=0,
                                                                                                          padx=20,
                                                                                                          pady=(10, 0),
                                                                                                          sticky="w")
        self.scale_var = ctk.StringVar(value="x4")
        self.scale_combo = ctk.CTkComboBox(self.sidebar, values=["x2", "x4"], variable=self.scale_var,
                                           command=self.change_model_scale)
        self.scale_combo.grid(row=5, column=0, padx=20, pady=(5, 10))

        # Face Enhance Switch
        self.use_face_enhance = ctk.BooleanVar(value=False)
        self.switch_face = ctk.CTkSwitch(self.sidebar, text="Face Enhancement", variable=self.use_face_enhance)
        self.switch_face.grid(row=6, column=0, padx=20, pady=10, sticky="w")

        self.btn_run = ctk.CTkButton(self.sidebar, text="🚀 Start Restoration", command=self.run_processing_thread,
                                     fg_color="#2CC985", hover_color="#229A66", height=50,
                                     font=ctk.CTkFont(size=16, weight="bold"))
        self.btn_run.grid(row=7, column=0, padx=20, pady=(20, 10))

        # Feature Visualization Button
        self.btn_vis = ctk.CTkButton(self.sidebar, text="🔍 Visualize Features", command=self.show_feature_maps,
                                     fg_color="#E0A800", hover_color="#B38600", height=40,
                                     font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_vis.grid(row=8, column=0, padx=20, pady=10)
        self.btn_vis.configure(state="disabled")

        self.btn_save = ctk.CTkButton(self.sidebar, text="💾 Save Result", command=self.save_result, state="disabled",
                                      height=40)
        self.btn_save.grid(row=9, column=0, padx=20, pady=10)

        # Metrics Display
        self.metrics_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.metrics_frame.grid(row=10, column=0, padx=20, pady=20, sticky="s")

        self.lbl_psnr = ctk.CTkLabel(self.metrics_frame, text="PSNR: --", font=ctk.CTkFont(size=16))
        self.lbl_psnr.pack(anchor="w")
        self.lbl_ssim = ctk.CTkLabel(self.metrics_frame, text="SSIM: --", font=ctk.CTkFont(size=16))
        self.lbl_ssim.pack(anchor="w")

        # === 2. Main Display Area (Right) ===
        self.display_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.display_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.display_frame.grid_columnconfigure((0, 1), weight=1)
        self.display_frame.grid_rowconfigure(1, weight=1)

        # Headers
        ctk.CTkLabel(self.display_frame, text="Original Input", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0,
                                                                                                               column=0,
                                                                                                               pady=5)
        ctk.CTkLabel(self.display_frame, text="Super-Resolution Output", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=1, pady=5)

        # Image Containers
        self.frame_input = ctk.CTkFrame(self.display_frame, fg_color=("gray90", "gray16"))
        self.frame_input.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.frame_output = ctk.CTkFrame(self.display_frame, fg_color=("gray90", "gray16"))
        self.frame_output.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # Image Labels
        self.lbl_img_in = ctk.CTkLabel(self.frame_input, text="Waiting for input...", corner_radius=6)
        self.lbl_img_in.pack(expand=True, fill="both", padx=10, pady=10)

        self.lbl_img_out = ctk.CTkLabel(self.frame_output, text="Waiting for processing...", corner_radius=6)
        self.lbl_img_out.pack(expand=True, fill="both", padx=10, pady=10)

        # === 3. Status Bar (Bottom) ===
        self.status_frame = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.status_frame.grid(row=2, column=1, sticky="ew")

        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", padx=10)
        self.status_label.pack(side="left")

        # Determinate Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.status_frame, width=300, mode="determinate")
        self.progress_bar.pack(side="right", padx=20, pady=5)
        self.progress_bar.set(0)

    # --- Feature Extraction Hook ---
    def hook_fn(self, module, input, output):
        self.feature_maps['conv_first'] = output.detach().cpu()

    # --- Logic: Progress Bar Animation ---
    def auto_increment_progress(self):
        if self.is_processing:
            current_val = self.progress_bar.get()
            if current_val < 0.95:
                increment = (0.95 - current_val) * 0.05
                if increment < 0.001: increment = 0.001
                self.progress_bar.set(current_val + increment)
            self.after(100, self.auto_increment_progress)

    # --- Logic: Model Loading & Switching ---
    def change_model_scale(self, choice):
        """Handle scale change event from combobox"""
        self.scale_factor = 2 if choice == "x2" else 4
        self.status_label.configure(text=f"Switching to {choice} model...")
        # Reload model in background to avoid freezing UI
        self.progress_bar.set(0.5)
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            # Determine which model file to use
            if self.scale_factor == 2:
                model_name = "RealESRGAN_x2plus.pth"
            else:
                model_name = "RealESRGAN_x4plus.pth"

            full_path = os.path.join(self.model_folder, model_name)

            if not os.path.exists(full_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_name}\nPlease download it to: {self.model_folder}")

            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32,
                                 scale=self.scale_factor)
            self.model.conv_first.register_forward_hook(self.hook_fn)

            self.upsampler = RealESRGANer(scale=self.scale_factor, model_path=full_path, model=self.model, tile=0,
                                          tile_pad=10, pre_pad=0, half=False, device=self.device)

            self.status_label.configure(text=f"✅ Model x{self.scale_factor} loaded | Device: {self.device}")
            self.progress_bar.set(0)
        except Exception as e:
            self.status_label.configure(text="❌ Load failed")
            messagebox.showerror("Model Error", str(e))

    def load_input_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.img_input = self.read_image(path)
            self.show_image_ctk(self.img_input, self.lbl_img_in)
            self.status_label.configure(text=f"Loaded: {os.path.basename(path)}")
            self.img_output = None
            self.lbl_img_out.configure(image=None, text="Waiting for processing...")
            self.btn_save.configure(state="disabled")
            self.btn_vis.configure(state="disabled")
            self.progress_bar.set(0)

    def load_gt_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.img_gt = self.read_image(path)
            messagebox.showinfo("Info", "Ground Truth loaded")

    def read_image(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def show_image_ctk(self, cv_img, label_widget):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)

        w_widget = label_widget.winfo_width()
        h_widget = label_widget.winfo_height()
        if w_widget < 10: w_widget = 500
        if h_widget < 10: h_widget = 500

        w_img, h_img = im_pil.size
        ratio = min(w_widget / w_img, h_widget / h_img)
        ratio = min(ratio, 1.0) if ratio > 1 else ratio

        new_w = int(w_img * ratio)
        new_h = int(h_img * ratio)

        ctk_img = ctk.CTkImage(light_image=im_pil, dark_image=im_pil, size=(new_w, new_h))

        label_widget.configure(image=ctk_img, text="")
        label_widget.image = ctk_img

    def run_processing_thread(self):
        if self.img_input is None: return
        self.btn_run.configure(state="disabled", text="⏳ Processing...")
        self.status_label.configure(text=f"Restoring image (x{self.scale_factor})...")

        self.is_processing = True
        self.progress_bar.set(0)
        self.auto_increment_progress()
        self.feature_maps = {}

        threading.Thread(target=self.process_image, daemon=True).start()

    def process_image(self):
        try:
            output = None
            used_face_enhance = False

            if self.use_face_enhance.get():
                try:
                    from gfpgan import GFPGANer
                    face_enhancer = GFPGANer(
                        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                        upscale=self.scale_factor, arch='clean', channel_multiplier=2, bg_upsampler=self.upsampler)
                    _, _, output = face_enhancer.enhance(self.img_input, has_aligned=False, only_center_face=False,
                                                         paste_back=True)
                    used_face_enhance = True
                except Exception as e:
                    print(f"Warning: Face enhance failed ({e}), switching to standard mode.")
                    self.after(0, lambda: self.status_label.configure(
                        text="⚠️ Face enhancement unavailable, switching to standard mode..."))

            if output is None:
                output, _ = self.upsampler.enhance(self.img_input, outscale=self.scale_factor)

            self.img_output = output

            self.after(0, lambda: self.show_image_ctk(self.img_output, self.lbl_img_out))
            self.after(0, self.calculate_metrics)

            if self.use_face_enhance.get() and not used_face_enhance:
                self.after(0, lambda: self.status_label.configure(text=f"✨ Done (x{self.scale_factor} Standard Mode)"))
            else:
                self.after(0, lambda: self.status_label.configure(text=f"✨ Done (x{self.scale_factor})"))

            self.after(0, lambda: self.btn_save.configure(state="normal"))
            if 'conv_first' in self.feature_maps:
                self.after(0, lambda: self.btn_vis.configure(state="normal"))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
        finally:
            self.is_processing = False
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self.btn_run.configure(state="normal", text="🚀 Start Restoration"))

    def calculate_metrics(self):
        if self.img_gt is None or self.img_output is None: return
        h, w = self.img_output.shape[:2]
        img_gt_aligned = cv2.resize(self.img_gt, (w, h))
        if psnr and ssim:
            s_psnr = psnr(img_gt_aligned, self.img_output, data_range=255)
            s_ssim = ssim(img_gt_aligned, self.img_output, data_range=255, channel_axis=2)
            self.lbl_psnr.configure(text=f"PSNR: {s_psnr:.2f} dB", text_color="#2CC985")
            self.lbl_ssim.configure(text=f"SSIM: {s_ssim:.4f}", text_color="#2CC985")

    def show_feature_maps(self):
        """Pop up a window to show feature maps"""
        if 'conv_first' not in self.feature_maps: return

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from tkinter import Toplevel

        feats = self.feature_maps['conv_first'][0]

        vis_window = Toplevel(self)
        vis_window.title("Intermediate Feature Maps")
        vis_window.geometry("800x800")

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f'Feature Maps (Scale x{self.scale_factor})', fontsize=16)

        for i, ax in enumerate(axes.flat):
            if i < feats.shape[0]:
                f_map = feats[i].numpy()
                ax.imshow(f_map, cmap='viridis')
            ax.axis('off')

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=vis_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def save_result(self):
        if self.img_output is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            cv2.imencode('.png', self.img_output)[1].tofile(path)
            messagebox.showinfo("Saved", "Image saved successfully")


if __name__ == "__main__":
    app = ModernApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
