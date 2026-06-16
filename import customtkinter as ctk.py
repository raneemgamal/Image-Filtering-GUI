import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import filters
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class DIPStudioUltra(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window & Engine Config ---
        self.title("DIP ENGINE | ULTRA PRO")
        self.after(0, lambda: self.state("zoomed"))
        ctk.set_appearance_mode("dark")
        
        # --- Cyber-Industrial Palette ---
        self.clr_black = "#050505"      
        self.clr_base = "#0D0D0D"       
        self.clr_panel = "#141414"      
        self.clr_accent = "#FFB800"     
        self.clr_neon = "#8B5CF6"       
        self.clr_border = "#222222"     
        self.clr_text = "#E0E0E0"
        self.font_mod = "Inter"         

        # --- Logic State ---
        self.undo_stack = []
        self.active_image = None
        self.source_image = None
        self.temp_buffer = None
        self.blend_img2 = None 
        self.current_mode = "ADJUST"

        self._setup_ui()

    def _setup_ui(self):
        self.configure(fg_color=self.clr_black)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. VERTICAL NAVIGATION DOCK
        self.nav_dock = ctk.CTkFrame(self, width=70, fg_color=self.clr_base, corner_radius=0, border_width=1, border_color=self.clr_border)
        self.nav_dock.grid(row=0, column=0, sticky="nsew")
        self.nav_dock.grid_propagate(False)

        self.create_dock_icon("⚙️", "ADJUST").pack(pady=(30, 10))
        self.create_dock_icon("🎨", "ARTISTIC").pack(pady=10)
        self.create_dock_icon("📐", "MORPH").pack(pady=10)
        self.create_dock_icon("🧪", "MATH").pack(pady=10)
        
        self.create_dock_icon("↩", "UNDO", self.execute_undo).pack(side="bottom", pady=10)
        self.create_dock_icon("🔄", "RESET", self.execute_reset).pack(side="bottom", pady=10)
        self.create_dock_icon("💾", "SAVE", self.export_file).pack(side="bottom", pady=10)
        self.create_dock_icon("📥", "OPEN", self.import_file).pack(side="bottom", pady=20)

        # 2. CONTROL PANEL
        self.control_panel = ctk.CTkFrame(self, width=280, fg_color=self.clr_panel, corner_radius=0, border_width=1, border_color=self.clr_border)
        self.control_panel.grid(row=0, column=1, sticky="nsew")
        
        self.panel_title = ctk.CTkLabel(self.control_panel, text="ENGINE / ADJUST", font=(self.font_mod, 14, "bold"), text_color=self.clr_accent)
        self.panel_title.pack(pady=25, padx=20, anchor="w")

        self.scroll_area = ctk.CTkScrollableFrame(self.control_panel, fg_color="transparent", scrollbar_button_color=self.clr_border)
        self.scroll_area.pack(fill="both", expand=True, padx=5)

        # 3. WORKSPACE
        self.workspace = ctk.CTkFrame(self, fg_color=self.clr_black, corner_radius=0)
        self.workspace.grid(row=0, column=2, sticky="nsew")
        
        self.viewport = ctk.CTkLabel(self.workspace, text="SYS_READY: STANDBY FOR INPUT", font=("Consolas", 14), text_color="#333")
        self.viewport.pack(expand=True, fill="both")

        # 4. HUD OVERLAY
        self.hud = ctk.CTkFrame(self.workspace, height=120, fg_color="#0D0D0D", corner_radius=15, border_width=1, border_color=self.clr_border)
        self.hud.place(relx=0.5, rely=0.92, anchor="center", relwidth=0.8)
        
        self.fig, self.ax = plt.subplots(figsize=(4, 1.0))
        self.fig.patch.set_facecolor("none")
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=self.hud)
        self.canvas_hist.get_tk_widget().pack(side="left", padx=20, pady=5)
        
        self.meta_lbl = ctk.CTkLabel(self.hud, text="DATA_LINK: OFFLINE\nBUFFER_2: EMPTY", font=("Consolas", 11), text_color=self.clr_accent, justify="right")
        self.meta_lbl.pack(side="right", padx=30)

        self.update_tool_ui()

    def create_dock_icon(self, icon, mode, cmd=None):
        def _set_mode():
            self.current_mode = mode
            self.update_tool_ui()
        callback = cmd if cmd else _set_mode
        return ctk.CTkButton(self.nav_dock, text=icon, width=50, height=50, fg_color="transparent", 
                            hover_color=self.clr_border, font=("Segoe UI", 20), command=callback)

    def create_slider_group(self, parent, label, min_v, max_v, start, cmd):
        f = ctk.CTkFrame(parent, fg_color="#1A1A1A", corner_radius=8, border_width=1, border_color=self.clr_border)
        f.pack(fill="x", pady=8, padx=10)
        lbl_row = ctk.CTkFrame(f, fg_color="transparent")
        lbl_row.pack(fill="x", padx=12, pady=(10, 0))
        ctk.CTkLabel(lbl_row, text=label, font=(self.font_mod, 11), text_color="#888").pack(side="left")
        v_lbl = ctk.CTkLabel(lbl_row, text=str(start), font=("Consolas", 11), text_color=self.clr_neon)
        v_lbl.pack(side="right")
        s = ctk.CTkSlider(f, from_=min_v, to=max_v, height=16, button_color=self.clr_neon, progress_color=self.clr_neon,
                         command=lambda v: self._on_slide(v, v_lbl, cmd))
        s.set(start)
        s.pack(fill="x", padx=12, pady=12)
        s.bind("<ButtonRelease-1>", self.finalize_action)

    def create_action_btn(self, parent, text, cmd, is_warn=False):
        clr = "#222" if not is_warn else "#301515"
        h_clr = self.clr_neon if not is_warn else "#E74C3C"
        btn = ctk.CTkButton(parent, text=text.upper(), command=cmd, fg_color=clr, hover_color=h_clr, 
                           height=35, font=(self.font_mod, 10, "bold"), anchor="w")
        btn.pack(fill="x", pady=4, padx=10)

    def _on_slide(self, val, lbl, cmd):
        lbl.configure(text=f"{val:.2f}" if isinstance(val, float) else str(int(val)))
        cmd(val)

    def update_tool_ui(self):
        for widget in self.scroll_area.winfo_children(): widget.destroy()
        self.panel_title.configure(text=f"ENGINE / {self.current_mode}")

        if self.current_mode == "ADJUST":
            self.create_slider_group(self.scroll_area, "Exposure", -100, 100, 0, self.update_brightness)
            self.create_slider_group(self.scroll_area, "Contrast", 0.5, 3.0, 1.0, self.update_contrast)
            
            ctk.CTkLabel(self.scroll_area, text="CHANNELS", font=(self.font_mod, 10, "bold"), text_color="#555").pack(pady=10)
            self.create_action_btn(self.scroll_area, "Extract Red", lambda: self.apply_rgb(filters.get_red_channel))
            self.create_action_btn(self.scroll_area, "Extract Green", lambda: self.apply_rgb(filters.get_green_channel))
            self.create_action_btn(self.scroll_area, "Extract Blue", lambda: self.apply_rgb(filters.get_blue_channel))
            
            ctk.CTkLabel(self.scroll_area, text="ENHANCEMENT", font=(self.font_mod, 10, "bold"), text_color="#555").pack(pady=10)
            self.create_action_btn(self.scroll_area, "Grayscale", lambda: self.apply_simple(filters.ensure_grayscale))
            self.create_action_btn(self.scroll_area, "Hist Stretching", lambda: self.apply_simple(filters.histogram_stretching_color))
            self.create_action_btn(self.scroll_area, "Floyd Dithering", lambda: self.apply_simple(filters.apply_floyd_steinberg))

        elif self.current_mode == "ARTISTIC":
            self.create_action_btn(self.scroll_area, "Highpass Sharp", lambda: self.apply_simple(filters.highpass_sharpen))
            self.create_action_btn(self.scroll_area, "Cinematic Warm", lambda: self.apply_simple(filters.cinematic_warm))
            self.create_action_btn(self.scroll_area, "Comic Style", lambda: self.apply_simple(filters.comic_filter))
            self.create_action_btn(self.scroll_area, "Sketch Effect", lambda: self.apply_simple(filters.sketch_filter))
            self.create_action_btn(self.scroll_area, "Dramatic FX", lambda: self.apply_simple(filters.dramatic_filter))
            self.create_action_btn(self.scroll_area, "Cool Filter", lambda: self.apply_simple(filters.cool_filter))

        elif self.current_mode == "MORPH":
            self.create_slider_group(self.scroll_area, "Mean Blur Radius", 1, 31, 1, self.update_mean)
            
            ctk.CTkLabel(self.scroll_area, text="OPERATIONS", font=(self.font_mod, 10, "bold"), text_color="#555").pack(pady=10)
            self.create_action_btn(self.scroll_area, "Erosion", lambda: self.apply_simple(filters.apply_erosion))
            self.create_action_btn(self.scroll_area, "Dilation", lambda: self.apply_simple(filters.apply_dilation))
            self.create_action_btn(self.scroll_area, "Opening", lambda: self.apply_simple(filters.apply_opening))
            self.create_action_btn(self.scroll_area, "Closing", lambda: self.apply_simple(filters.apply_closing))
            self.create_action_btn(self.scroll_area, "Min Filter", lambda: self.apply_simple(filters.apply_min_filter))
            self.create_action_btn(self.scroll_area, "Max Filter", lambda: self.apply_simple(filters.apply_max_filter))
            
            ctk.CTkLabel(self.scroll_area, text="NOISE CONTROL", font=(self.font_mod, 10, "bold"), text_color="#555").pack(pady=10)
            self.create_action_btn(self.scroll_area, "Add Salt & Pepper", lambda: self.apply_simple(filters.add_salt_and_pepper))
            self.create_action_btn(self.scroll_area, "Median Restore", lambda: self.apply_simple(filters.restore_image))

        elif self.current_mode == "MATH":
            self.create_action_btn(self.scroll_area, "Load Buffer Image", self.load_second_buffer)
            self.create_slider_group(self.scroll_area, "Alpha Blend", 0, 1, 0.5, self.update_blend)
            
            ctk.CTkLabel(self.scroll_area, text="LINEAR MATH", font=(self.font_mod, 10, "bold"), text_color="#555").pack(pady=10)
            self.create_action_btn(self.scroll_area, "Linear Add", lambda: self.proc_math("add"))
            self.create_action_btn(self.scroll_area, "Linear Subtract", lambda: self.proc_math("sub"))
            self.create_action_btn(self.scroll_area, "Difference Map", lambda: self.proc_math("diff"))

    def show_image(self, img, is_preview=False):
        if is_preview: self.last_preview = img
        render_img = np.ascontiguousarray(np.uint8(np.clip(img, 0, 255)))
        
        # Meta Update
        b2_status = "READY" if self.blend_img2 is not None else "EMPTY"
        self.meta_lbl.configure(text=f"STATUS: ONLINE\nRES: {render_img.shape[1]}x{render_img.shape[0]}px\nBUFFER_2: {b2_status}")

        self.ax.clear()
        self.ax.set_facecolor("none")
        if render_img.ndim == 3:
            # FIX: Mapping OpenCV BGR order to Histogram colors
            # OpenCV BGR: 0=Blue, 1=Green, 2=Red
            hist_colors = ("#0000FF", "#00FF00", "#FF0000") 
            for i, col in enumerate(hist_colors):
                h = cv2.calcHist([render_img], [i], None, [256], [0, 256])
                self.ax.plot(h.flatten(), color=col, lw=1.2, alpha=0.8)
            rgb = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
        else:
            h = cv2.calcHist([render_img], [0], None, [256], [0, 256])
            self.ax.plot(h.flatten(), color="#E0E0E0", lw=1.2) # Gray for grayscale
            rgb = cv2.cvtColor(render_img, cv2.COLOR_GRAY2RGB)
        
        self.ax.set_xlim([0, 256])
        self.ax.axis("off")
        self.canvas_hist.draw()

        pil = Image.fromarray(rgb)
        w, h = self.workspace.winfo_width(), self.workspace.winfo_height()
        if w < 10: w, h = 1200, 800
        ratio = min((w-100) / pil.width, (h-150) / pil.height)
        resized = pil.resize((int(pil.width * ratio), int(pil.height * ratio)), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        self.viewport.configure(image=tk_img, text="")
        self.viewport.image = tk_img

    def import_file(self):
        p = filedialog.askopenfilename()
        if p:
            self.active_image = filters.load_image_from_disk(p)
            self.source_image = self.active_image.copy()
            self.undo_stack.clear()
            self.show_image(self.active_image)

    def load_second_buffer(self):
        p = filedialog.askopenfilename()
        if p: 
            self.blend_img2 = filters.load_image_from_disk(p)
            messagebox.showinfo("Buffer", "Secondary Image Loaded.")
            self.show_image(self.active_image)

    def push_undo(self):
        if self.active_image is not None: self.undo_stack.append(self.active_image.copy())

    def apply_simple(self, func):
        if self.active_image is not None:
            self.push_undo()
            self.active_image = func(self.active_image)
            self.show_image(self.active_image)

    def apply_rgb(self, func):
        if self.source_image is not None:
            self.push_undo()
            self.active_image = func(self.source_image.copy())
            self.show_image(self.active_image)

    def update_brightness(self, v):
        if self.active_image is not None:
            self._prepare_preview()
            self.show_image(filters.adjust_brightness(self.temp_buffer, int(v)), True)

    def update_contrast(self, v):
        if self.active_image is not None:
            self._prepare_preview()
            self.show_image(filters.adjust_contrast(self.temp_buffer, float(v)), True)

    def update_mean(self, v):
        if self.active_image is not None:
            self._prepare_preview()
            self.show_image(filters.apply_mean_filter(self.temp_buffer, int(v)|1), True)

    def update_blend(self, v):
        if self.active_image is not None and self.blend_img2 is not None:
            self._prepare_preview()
            self.show_image(filters.blend_images(self.temp_buffer, self.blend_img2, float(v)), True)

    def proc_math(self, op):
        if self.active_image is None or self.blend_img2 is None:
            messagebox.showwarning("Buffer Error", "Please load a Second Image first.")
            return
        self.push_undo()
        if op == "add": self.active_image = filters.add_images(self.active_image, self.blend_img2)
        elif op == "sub": self.active_image = filters.subtract_images(self.active_image, self.blend_img2)
        elif op == "diff": self.active_image = filters.diff_images(self.active_image, self.blend_img2)
        self.show_image(self.active_image)

    def _prepare_preview(self):
        if self.temp_buffer is None: self.temp_buffer = self.active_image.copy()

    def finalize_action(self, _):
        if self.temp_buffer is not None:
            self.push_undo()
            self.active_image = self.last_preview.copy()
            self.temp_buffer = None

    def execute_undo(self):
        if self.undo_stack:
            self.active_image = self.undo_stack.pop()
            self.show_image(self.active_image)

    def execute_reset(self):
        if self.source_image is not None:
            self.push_undo()
            self.active_image = self.source_image.copy()
            self.show_image(self.active_image)

    def export_file(self):
        if self.active_image is not None:
            p = filedialog.asksaveasfilename(defaultextension=".jpg")
            if p: filters.save_image_to_disk(p, self.active_image)

if __name__ == "__main__":
    app = DIPStudioUltra()
    app.mainloop()