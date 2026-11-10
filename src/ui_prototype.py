from __future__ import annotations

import ctypes
from datetime import datetime
import math
import queue
import sys
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import font as tkfont
from tkinter import filedialog, messagebox, ttk

try:
    import customtkinter as ctk
except ImportError as error:  # pragma: no cover - depends on the user's environment
    print(
        "CustomTkinter is required for this visual prototype. Install it in the "
        "project virtual environment, then run src\\ui_prototype.py.",
        file=sys.stderr,
    )
    raise SystemExit(1) from error


NAVY = "#183B56"
NAVY_DARK = "#102B40"
TEAL = "#087F8C"
PURPLE = "#6D4AFF"
GREEN = "#287A48"
RED = "#B64242"
AMBER = "#946200"
BACKGROUND = "#F2F5F7"
CARD = "#FFFFFF"
TEXT = "#20313D"
MUTED = "#687985"
LINE = "#DCE4E9"
GREEN_SOFT = "#EAF4ED"
RED_SOFT = "#FAEEEE"
PREFERRED_FONT_FAMILY = "Segoe UI"
FONT_FALLBACKS = ("Arial", "Liberation Sans", "DejaVu Sans", "Helvetica")
APP_USER_MODEL_ID = "KI_Qualitaetspruefung.UIPrototype"
CLASS_NAMES = ("gebrochen", "intakt")
TEST_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TRAIN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


class QualityUiPrototype:
    """A presentation-only desktop layout with clearly marked sample content."""

    def __init__(self) -> None:
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.project_root = Path(__file__).resolve().parent.parent
        self.logo_path = self.project_root / "assets" / "qualitaetspruefung.png"
        self.admin_icon_path = self.project_root / "assets" / "admin.png"
        self.window_icon_path = self.project_root / "assets" / "qualitaetspruefung.ico"
        self.logo_image: ctk.CTkImage | None = None
        self.admin_icon_image: ctk.CTkImage | None = None
        self._admin_authenticated = False
        self._session_token: str | None = None
        self._comparison_authorized = False
        self._admin_login_window: ctk.CTkToplevel | None = None
        self._admin_window: ctk.CTkToplevel | None = None
        self._set_app_user_model_id()
        self.root = ctk.CTk()
        self._set_windows_icon()
        self.font_family = self._select_font_family()
        self.root.title("Qualitätsprüfung")
        self.root.geometry("1360x920")
        self.root.minsize(1040, 740)
        self.root.configure(fg_color=BACKGROUND)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._training_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._training_active = False
        self._training_poll_id: str | None = None
        self.trained_model_path: Path | None = None
        self._evaluation_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._evaluation_active = False
        self._evaluation_poll_id: str | None = None
        self.test_folder: Path | None = None
        self.test_model_path: Path | None = None
        self.test_records: list[dict[str, object]] = []
        self.completed_evaluation: dict[str, object] | None = None
        self._load_header_logo()
        self._load_admin_icon()

        try:
            self._configure_table_style()
            self._build_header()
            self._build_tabs()
            self._poll_training_queue()
        except Exception:
            self._close()
            raise

    def _select_font_family(self) -> str:
        """Use Segoe UI when installed, otherwise choose an available sans-serif."""
        try:
            available = {family.casefold(): family for family in tkfont.families(self.root)}
        except tk.TclError:
            available = {}
        for candidate in (PREFERRED_FONT_FAMILY, *FONT_FALLBACKS):
            if candidate.casefold() in available:
                return available[candidate.casefold()]
        try:
            return str(tkfont.nametofont("TkDefaultFont", root=self.root).actual("family"))
        except tk.TclError:
            return "TkDefaultFont"

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self._close()

    @staticmethod
    def _set_app_user_model_id() -> None:
        """Give Windows a stable app identity for taskbar grouping when supported."""
        if sys.platform != "win32":
            return
        try:
            set_app_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
            set_app_id.argtypes = [ctypes.c_wchar_p]
            set_app_id.restype = ctypes.c_long
            set_app_id(APP_USER_MODEL_ID)
        except (AttributeError, OSError, TypeError, ValueError, ctypes.ArgumentError):
            pass

    def _load_header_logo(self) -> None:
        """Load the optional project PNG into a retained CustomTkinter image."""
        if not self.logo_path.is_file():
            warnings.warn(
                f"Header logo not found at {self.logo_path}; the header logo will be omitted.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        try:
            from PIL import Image

            resampling = getattr(Image, "Resampling", Image).LANCZOS
            with Image.open(self.logo_path) as source:
                image = source.convert("RGBA")
            scale = min(48 / image.width, 48 / image.height)
            display_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(display_size, resampling)
            self.logo_image = ctk.CTkImage(light_image=image, dark_image=image.copy(), size=display_size)
        except Exception as error:
            warnings.warn(
                f"Header logo at {self.logo_path} could not be loaded ({error}); "
                "the header logo will be omitted.", RuntimeWarning, stacklevel=2,
            )

    def _load_admin_icon(self) -> None:
        """Load the provided Admin profile icon without depending on cwd."""
        if not self.admin_icon_path.is_file():
            warnings.warn(f"Admin icon not found at {self.admin_icon_path}.", RuntimeWarning, stacklevel=2)
            return
        try:
            from PIL import Image

            with Image.open(self.admin_icon_path) as source:
                image = source.convert("RGBA")
            self.admin_icon_image = ctk.CTkImage(
                light_image=image,
                dark_image=image.copy(),
                size=(34, 34),
            )
        except Exception as error:
            warnings.warn(f"Admin icon could not be loaded ({error}).", RuntimeWarning, stacklevel=2)

    def _set_windows_icon(self) -> None:
        """Apply the optional project ICO on Windows without generating files."""
        if sys.platform != "win32":
            return
        if not self.window_icon_path.is_file():
            warnings.warn(
                f"Optional Windows icon not found at {self.window_icon_path}; "
                "continuing without a custom window icon.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        try:
            self.root.iconbitmap(str(self.window_icon_path))
        except (OSError, tk.TclError, AttributeError) as error:
            warnings.warn(
                f"Windows could not load the icon at {self.window_icon_path} ({error}); "
                "continuing with the default window icon.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _close(self) -> None:
        """Keep the window open until training and evaluation workers finish."""
        if getattr(self, "_training_active", False) or getattr(self, "_evaluation_active", False):
            messagebox.showinfo(
                "Vorgang läuft",
                "Bitte warten Sie, bis Training oder Auswertung abgeschlossen ist, bevor Sie die Anwendung schliessen.",
                parent=self.root,
            )
            return
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _open_admin_login(self) -> None:
        if self._admin_authenticated:
            self._open_admin_panel()
            return
        if self._admin_login_window is not None and self._admin_login_window.winfo_exists():
            self._admin_login_window.focus_force()
            return

        window = ctk.CTkToplevel(self.root)
        self._admin_login_window = window
        window.title("Admin")
        window.geometry("390x250")
        window.resizable(False, False)
        window.configure(fg_color=BACKGROUND)
        window.transient(self.root)
        window.grid_columnconfigure(0, weight=1)
        card = self._card(window)
        card.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        ctk.CTkLabel(
            card, text="Admin-Zugang", text_color=NAVY_DARK,
            font=ctk.CTkFont(family=self.font_family, size=21, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(17, 10))
        token_entry = ctk.CTkEntry(
            card, placeholder_text="Zugangscode eingeben", show="*", height=40,
            font=ctk.CTkFont(family=self.font_family, size=14),
        )
        token_entry.pack(fill="x", padx=18, pady=(0, 8))
        error_label = ctk.CTkLabel(
            card, text="", text_color=RED, font=ctk.CTkFont(family=self.font_family, size=12),
        )
        error_label.pack(anchor="w", padx=18, pady=(0, 5))

        def unlock() -> None:
            token = token_entry.get().strip()
            if token != "KIQL26#" and (self._session_token is None or token != self._session_token):
                token_entry.delete(0, "end")
                error_label.configure(text="Ungültiger Zugangscode.")
                return
            self._admin_authenticated = True
            self._render_protected_metrics()
            self._refresh_evaluation_export_buttons()
            window.destroy()
            self._admin_login_window = None
            self._open_admin_panel()

        ctk.CTkButton(
            card, text="Admin-Zugang freischalten", height=40, fg_color=NAVY, hover_color=NAVY_DARK,
            font=ctk.CTkFont(family=self.font_family, size=14, weight="bold"), command=unlock,
        ).pack(fill="x", padx=18, pady=(2, 16))
        token_entry.bind("<Return>", lambda _event: unlock())
        window.protocol("WM_DELETE_WINDOW", lambda: self._destroy_admin_login(window))
        token_entry.focus_set()

    def _destroy_admin_login(self, window: ctk.CTkToplevel) -> None:
        try:
            window.destroy()
        except tk.TclError:
            pass
        if self._admin_login_window is window:
            self._admin_login_window = None

    def _open_admin_panel(self) -> None:
        if self._admin_window is not None and self._admin_window.winfo_exists():
            self._admin_window.focus_force()
            return
        window = ctk.CTkToplevel(self.root)
        self._admin_window = window
        window.title("Admin")
        window.geometry("460x390")
        window.resizable(False, False)
        window.configure(fg_color=BACKGROUND)
        window.transient(self.root)
        card = self._card(window)
        card.pack(fill="both", expand=True, padx=20, pady=20)
        ctk.CTkLabel(
            card, text="Admin-Zugang", text_color=NAVY_DARK,
            font=ctk.CTkFont(family=self.font_family, size=21, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(17, 3))
        ctk.CTkLabel(
            card, text="Sitzungtoken", text_color=NAVY,
            font=ctk.CTkFont(family=self.font_family, size=16, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(10, 8))
        new_token = ctk.CTkEntry(card, placeholder_text="Neues temporäres Token", show="*", height=40)
        new_token.pack(fill="x", padx=18, pady=5)
        confirm_token = ctk.CTkEntry(card, placeholder_text="Temporäres Token bestätigen", show="*", height=40)
        confirm_token.pack(fill="x", padx=18, pady=5)
        feedback = ctk.CTkLabel(card, text="", text_color=MUTED)
        feedback.pack(anchor="w", padx=18, pady=(4, 3))

        def save_token() -> None:
            if not self._admin_authenticated:
                feedback.configure(text="Admin-Authentifizierung erforderlich.", text_color=RED)
                return
            first, second = new_token.get().strip(), confirm_token.get().strip()
            if not first or not second:
                feedback.configure(text="Tokenfelder dürfen nicht leer sein.", text_color=RED)
            elif first != second:
                feedback.configure(text="Die Token stimmen nicht überein.", text_color=RED)
            else:
                self._session_token = first
                self._comparison_authorized = False
                new_token.delete(0, "end")
                confirm_token.delete(0, "end")
                feedback.configure(text="Temporärer Token wurde für diese Sitzung gespeichert.", text_color=GREEN)
                self._render_protected_metrics()

        ctk.CTkButton(
            card, text="Temporären Token speichern", height=40, fg_color=TEAL, hover_color="#066C77",
            command=save_token,
        ).pack(fill="x", padx=18, pady=(5, 8))
        ctk.CTkButton(
            card, text="Admin-Zugang sperren", height=38, fg_color="#EDF2F5", hover_color="#DDE7EC",
            text_color=NAVY, command=self._lock_admin,
        ).pack(fill="x", padx=18, pady=(0, 16))
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_admin_panel(window))

    def _close_admin_panel(self, window: ctk.CTkToplevel) -> None:
        try:
            window.destroy()
        except tk.TclError:
            pass
        if self._admin_window is window:
            self._admin_window = None

    def _lock_admin(self) -> None:
        self._admin_authenticated = False
        self._render_protected_metrics()
        self._refresh_evaluation_export_buttons()
        if self._admin_window is not None:
            self._close_admin_panel(self._admin_window)

    def _render_protected_metrics(self) -> None:
        result = self.completed_evaluation
        metrics = result.get("metrics") if result and self._has_reference_access() else None
        if metrics is None:
            for label in self.metric_value_labels.values():
                label.configure(text="--")
            for label in self.confusion_matrix_values.values():
                label.configure(text="--")
            for label in self.rate_value_labels.values():
                label.configure(text="--")
            for label in self.reference_count_labels.values():
                label.configure(text="--")
            return
        matrix = metrics["confusion_matrix"]
        for true_index in range(2):
            for predicted_index in range(2):
                self.confusion_matrix_values[(true_index, predicted_index)].configure(
                    text=str(matrix[true_index][predicted_index])
                )
        for key in ("accuracy", "sensitivity", "specificity"):
            label = self.metric_value_labels[key]
            value = metrics[key]
            label.configure(text="N/A" if value is None else f"{float(value) * 100:.2f}%")
        # Gebrochen is the positive class (matrix index 0), matching Sensitivität.
        tp, fn = matrix[0][0], matrix[0][1]
        fp, tn = matrix[1][0], matrix[1][1]
        for key, numerator, denominator in (
            ("fpr", fp, tn + fp),
            ("fnr", fn, tp + fn),
        ):
            value = "-- %" if denominator == 0 else f"{numerator / denominator * 100:.2f}%"
            self.metric_value_labels[key].configure(text=value)
        for key, label in self.rate_value_labels.items():
            label.configure(text=self._format_evaluation_metric(metrics[key]))
        for key, label in self.reference_count_labels.items():
            label.configure(text=str(metrics[key]))

    def _has_reference_access(self) -> bool:
        return self._admin_authenticated or self._comparison_authorized

    def _compare_student_results(self) -> None:
        self._comparison_authorized = False
        self._refresh_evaluation_export_buttons()
        if not self._session_token:
            messagebox.showinfo(
                "Vergleich nicht verfügbar",
                "Es wurde noch kein Sitzungstoken eingerichtet.",
                parent=self.root,
            )
            self._render_protected_metrics()
            return
        if self.completed_evaluation is None:
            messagebox.showinfo(
                "Auswertung erforderlich",
                "Bitte zuerst den Test auswerten.",
                parent=self.root,
            )
            self._render_protected_metrics()
            return
        candidate = self.student_comparison_entry.get().strip()
        self.student_comparison_entry.delete(0, "end")
        if candidate != self._session_token:
            messagebox.showerror(
                "Token ungültig",
                "Die Referenzergebnisse bleiben gesperrt.",
                parent=self.root,
            )
            self._render_protected_metrics()
            return
        self._comparison_authorized = True
        self._refresh_evaluation_export_buttons()
        messagebox.showinfo(
            "Vergleich freigeschaltet",
            "Sie können Ihre Excel-Ergebnisse jetzt mit den Referenzwerten vergleichen.",
            parent=self.root,
        )
        self._render_protected_metrics()

    def _configure_table_style(self) -> None:
        style = ttk.Style(self.root)
        style.configure(
            "Prototype.Treeview",
            background=CARD,
            fieldbackground=CARD,
            foreground=TEXT,
            rowheight=34,
            font=(self.font_family, 12),
            borderwidth=0,
        )
        style.configure(
            "Prototype.Treeview.Heading",
            background="#EAF0F3",
            foreground=NAVY_DARK,
            padding=(9, 7),
            font=(self.font_family, 13, "bold"),
        )
        style.map(
            "Prototype.Treeview",
            background=[("selected", "#DCEAF0")],
            foreground=[("selected", TEXT)],
        )

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self.root, fg_color=CARD, corner_radius=14, border_width=1, border_color=LINE)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 14))
        header.grid_columnconfigure(1, weight=1)

        mark = ctk.CTkFrame(header, width=48, height=48, fg_color="transparent", corner_radius=0)
        mark.grid(row=0, column=0, rowspan=2, padx=(18, 14), pady=15)
        mark.grid_propagate(False)
        if self.logo_image is not None:
            ctk.CTkLabel(mark, image=self.logo_image, text="", fg_color="transparent").place(
                relx=0.5, rely=0.5, anchor="center"
            )

        ctk.CTkLabel(
            header,
            text="KI in der Qualitätsprüfung",
            text_color=NAVY_DARK,
            font=ctk.CTkFont(family=self.font_family, size=22, weight="bold"),
            anchor="w",
        ).grid(row=0, column=1, sticky="sw", pady=(13, 0))
        ctk.CTkLabel(
            header,
            text="KI-basierte Bauteilprüfung",
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=14),
            anchor="w",
        ).grid(row=1, column=1, sticky="nw", pady=(1, 13))

        status = ctk.CTkFrame(header, fg_color="#EAF4ED", corner_radius=14)
        status.grid(row=0, column=2, rowspan=2, padx=18, pady=17, sticky="e")
        ready_dot = tk.Canvas(status, width=14, height=14, background="#EAF4ED", highlightthickness=0)
        ready_dot.pack(side="left", padx=(13, 6), pady=9)
        ready_dot.create_oval(2, 2, 12, 12, fill=GREEN, outline=GREEN)
        ctk.CTkLabel(
            status,
            text="Programm Bereit",
            text_color=GREEN,
            font=ctk.CTkFont(family=self.font_family, size=13, weight="bold"),
        ).pack(side="left", padx=(0, 13), pady=8)

        if self.admin_icon_image is not None:
            ctk.CTkButton(
                header,
                text="",
                image=self.admin_icon_image,
                width=38,
                height=38,
                fg_color="transparent",
                hover_color="#E8EEF1",
                corner_radius=19,
                command=self._open_admin_login,
            ).grid(row=0, column=3, rowspan=2, padx=(0, 16), pady=17, sticky="e")

    def _build_tabs(self) -> None:
        self.tabs = ctk.CTkTabview(
            self.root,
            corner_radius=14,
            fg_color=CARD,
            segmented_button_fg_color="#E8EEF1",
            segmented_button_unselected_color="#E8EEF1",
            segmented_button_unselected_hover_color="#DDE7EC",
            segmented_button_selected_color="#D9E7EF",
            segmented_button_selected_hover_color="#C9DCE7",
            text_color=TEXT,
            segmented_button_font=ctk.CTkFont(family=self.font_family, size=14, weight="bold"),
            command=self._tab_changed,
        )
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 22))
        self.tabs.add("Training")
        self.tabs.add("Test & Ergebnisse")
        self.tabs.add("Export")
        self._build_training_tab(self.tabs.tab("Training"))
        self._build_testing_tab(self.tabs.tab("Test & Ergebnisse"))
        self._build_export_tab(self.tabs.tab("Export"))

    def _tab_changed(self) -> None:
        selected_fill = {
            "Training": "#D9E7EF",
            "Test & Ergebnisse": "#D8EEEE",
            "Export": "#E9E3FF",
        }.get(self.tabs.get(), "#D9E7EF")
        hover_fill = {
            "Training": "#C9DCE7",
            "Test & Ergebnisse": "#C5E2E3",
            "Export": "#DCD3FF",
        }.get(self.tabs.get(), "#C9DCE7")
        self.tabs.configure(
            segmented_button_selected_color=selected_fill,
            segmented_button_selected_hover_color=hover_fill,
        )

    @staticmethod
    def _draw_icon(canvas: tk.Canvas, kind: str, color: str) -> None:
        """Draw small line icons with Tk Canvas to avoid external icon assets."""
        if kind == "test":
            canvas.create_oval(3, 3, 13, 13, outline=color, width=2)
            canvas.create_line(12, 12, 19, 19, fill=color, width=2, capstyle="round")
        elif kind == "training":
            canvas.create_rectangle(3, 4, 17, 17, outline=color, width=1.7)
            canvas.create_line(6, 8, 14, 8, fill=color, width=1.5)
            canvas.create_line(6, 11, 11, 11, fill=color, width=1.5)
            canvas.create_line(6, 14, 13, 14, fill=color, width=1.5)
        elif kind == "export":
            canvas.create_rectangle(4, 3, 14, 17, outline=color, width=1.7)
            canvas.create_line(7, 7, 12, 7, fill=color, width=1.3)
            canvas.create_line(7, 10, 11, 10, fill=color, width=1.3)
            canvas.create_line(11, 15, 18, 15, fill=color, width=1.7, arrow="last")
        elif kind == "folder":
            canvas.create_line(2, 6, 7, 6, 9, 8, 18, 8, 18, 16, 2, 16, 2, 6,
                              fill=color, width=1.7, joinstyle="round")
            canvas.create_line(2, 9, 18, 9, fill=color, width=1.2)

    def _heading(self, parent: ctk.CTkFrame, text: str, color: str = NAVY, icon: str = "training") -> ctk.CTkFrame:
        heading = ctk.CTkFrame(parent, fg_color="transparent")
        canvas = tk.Canvas(heading, width=24, height=24, background=CARD, highlightthickness=0)
        canvas.pack(side="left", padx=(0, 8))
        self._draw_icon(canvas, icon, color)
        ctk.CTkLabel(
            heading,
            text=text,
            text_color=color,
            font=ctk.CTkFont(family=self.font_family, size=17, weight="bold"),
            anchor="w",
        ).pack(side="left")
        return heading

    def _description(self, parent: ctk.CTkFrame, text: str, color: str = MUTED) -> ctk.CTkLabel:
        label = ctk.CTkLabel(
            parent,
            text=text,
            text_color=color,
            font=ctk.CTkFont(family=self.font_family, size=14),
            justify="left",
            anchor="w",
            wraplength=900,
        )

        def resize(event: tk.Event) -> None:
            label.configure(wraplength=max(240, event.width - 42))

        parent.bind("<Configure>", resize, add="+")
        return label

    def _card(self, parent: ctk.CTkFrame, **kwargs) -> ctk.CTkFrame:
        return ctk.CTkFrame(
            parent,
            fg_color=CARD,
            corner_radius=12,
            border_width=1,
            border_color=LINE,
            **kwargs,
        )

    def _build_training_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure((0, 1), weight=1, uniform="training_cards")
        tab.grid_rowconfigure(5, weight=1)

        self.training_folders: dict[str, Path | None] = {"gebrochen": None, "intakt": None}
        self.training_folder_labels: dict[str, ctk.CTkLabel] = {}
        self.training_folder_buttons: dict[str, ctk.CTkButton] = {}
        self.training_entries: dict[str, ctk.CTkEntry] = {}
        self.training_variables: dict[str, tk.StringVar] = {}

        self._heading(tab, "Training", NAVY, "training").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(18, 3)
        )
        self._description(
            tab,
            "Ordnen Sie die Datensätze den beiden Qualitätsklassen zu. "
            "Das Training verwendet einen reproduzierbaren 80-20% Split je Klasse.",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 14))

        dataset_card = self._card(tab)
        dataset_card.grid(row=2, column=0, sticky="nsew", padx=(18, 8), pady=(0, 12))
        dataset_card.grid_columnconfigure(1, weight=1)
        self._card_heading(dataset_card, "Datensatzordner", NAVY).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(13, 12)
        )
        self._folder_row(dataset_card, 1, "gebrochen", NAVY)
        self._folder_row(dataset_card, 2, "intakt", NAVY)
        self.training_summary = ctk.CTkLabel(
            dataset_card,
            text="Datensatzanzahl: -   ",
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=13),
            anchor="w",
            justify="left",
        )
        self.training_summary.grid(row=3, column=0, columnspan=3, sticky="ew", padx=15, pady=(10, 14))
        self._bind_wraplength(self.training_summary, dataset_card, padding=42)

        parameters = self._card(tab)
        parameters.grid(row=2, column=1, sticky="nsew", padx=(8, 18), pady=(0, 12))
        parameters.grid_columnconfigure(1, weight=1)
        self._card_heading(parameters, "Trainingsparameter", NAVY).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(13, 10)
        )
        for row, label, value in (
            (1, "Epochen", "30"),
            (2, "Batchgröße", "32"),
            (3, "Lernrate", "0.001"),
        ):
            ctk.CTkLabel(
                parameters,
                text=label,
                text_color=TEXT,
                font=ctk.CTkFont(family=self.font_family, size=14),
                anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=15, pady=5)
            variable = tk.StringVar(value=value)
            entry = ctk.CTkEntry(
                parameters,
                textvariable=variable,
                height=40,
                font=ctk.CTkFont(family=self.font_family, size=14),
            )
            entry.grid(row=row, column=1, sticky="ew", padx=(8, 15), pady=5)
            self.training_entries[label] = entry
            self.training_variables[label] = variable

        run_row = ctk.CTkFrame(tab, fg_color="transparent")
        run_row.grid(row=3, column=0, columnspan=2, sticky="new", padx=18, pady=(2, 8))
        self.train_button = ctk.CTkButton(
            run_row,
            text="Training starten",
            width=166,
            height=44,
            fg_color=NAVY,
            hover_color=NAVY_DARK,
            font=ctk.CTkFont(family=self.font_family, size=14, weight="bold"),
            command=self._start_training,
        )
        self.train_button.pack(side="left")
        self.output_button = ctk.CTkButton(
            run_row,
            text="Modellpfad wählen",
            width=158,
            height=44,
            fg_color="#EDF2F5",
            hover_color="#DDE7EC",
            text_color=NAVY,
            font=ctk.CTkFont(family=self.font_family, size=14),
            command=self._select_model_output,
        )
        self.output_button.pack(side="left", padx=(10, 0))
        self.training_action_status = ctk.CTkLabel(
            run_row,
            text="Ausgabepfad erforderlich",
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=13),
            anchor="w",
        )
        self.training_action_status.pack(side="left", padx=13, fill="x", expand=True)

        progress = self._card(tab)
        progress.grid(row=4, column=0, columnspan=2, sticky="new", padx=18, pady=(0, 10))
        progress.grid_columnconfigure(0, weight=1)
        self.training_progress = ctk.CTkProgressBar(
            progress, height=10, progress_color=NAVY, fg_color="#E7EDF0"
        )
        self.training_progress.set(0)
        self.training_progress.grid(row=0, column=0, sticky="ew", padx=15, pady=(13, 5))
       
        self.model_path_status = ctk.CTkLabel(
            progress,
            text="Kein Modell-Ausgabepfad ausgewählt",
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=12),
            anchor="w",
        )
        self.model_path_status.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 12))

        history = self._card(tab)
        history.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=18, pady=(0, 14))
        history.grid_columnconfigure(0, weight=1)
        self._card_heading(history, "Trainingsverlauf", NAVY).grid(
            row=0, column=0, sticky="w", padx=15, pady=(12, 3)
        )
        self.training_history = ctk.CTkTextbox(
            history,
            height=132,
            fg_color=CARD,
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=13),
            corner_radius=0,
            border_width=0,
            scrollbar_button_color=LINE,
            scrollbar_button_hover_color="#C8D4DC",
            activate_scrollbars=True,
            wrap="word",
        )
        self.training_history.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 12))
        self.training_history.insert("1.0", "Noch keine Trainingsdaten.")
        self.training_history.configure(state="disabled")
        self._training_controls = [
            *self.training_folder_buttons.values(),
            *self.training_entries.values(),
            self.output_button,
            self.train_button,
        ]
        self._training_history_epoch_count = 0
        self._selected_model_output: Path | None = None

    def _folder_row(self, parent: ctk.CTkFrame, row: int, class_name: str, accent: str) -> None:
        ctk.CTkLabel(
            parent,
            text=class_name.capitalize(),
            width=92,
            text_color=TEXT,
            font=ctk.CTkFont(family=self.font_family, size=14, weight="bold"),
            anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(15, 5), pady=5)
        selected_path = ctk.CTkLabel(
            parent,
            text="Noch kein Ordner ausgewählt",
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=13),
            anchor="w",
        )
        selected_path.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.training_folder_labels[class_name] = selected_path
        action = ctk.CTkFrame(parent, fg_color="transparent")
        action.grid(row=row, column=2, sticky="e", padx=(5, 15), pady=5)
        icon = tk.Canvas(action, width=20, height=20, background=CARD, highlightthickness=0)
        icon.pack(side="left", padx=(0, 4))
        self._draw_icon(icon, "folder", accent)
        button = ctk.CTkButton(
            action,
            text="Ordner wählen",
            width=128,
            height=40,
            fg_color="#EDF2F5",
            hover_color="#DDE7EC",
            text_color=accent,
            font=ctk.CTkFont(family=self.font_family, size=14),
            command=lambda name=class_name: self._select_training_folder(name),
        )
        button.pack(side="left")
        self.training_folder_buttons[class_name] = button

    def _card_heading(self, parent: ctk.CTkFrame, text: str, color: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent,
            text=text,
            text_color=color,
            font=ctk.CTkFont(family=self.font_family, size=17, weight="bold"),
            anchor="w",
        )

    def _build_testing_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        scrollable_tab = ctk.CTkScrollableFrame(tab, fg_color="transparent", corner_radius=0)
        scrollable_tab.grid(row=0, column=0, sticky="nsew")
        scrollable_tab.grid_columnconfigure((0, 1, 2), weight=1, uniform="test_metrics")
        tab = scrollable_tab
        self._heading(tab, "Auswertung", TEAL, "test").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=18, pady=(16, 2)
        )
        self._description(
            tab,
            "Wahre Labels werden aus den Klassenunterordnern „gebrochen“ und „intakt“ übernommen. "
            "Der Testdatensatz bleibt von Training und Validierung getrennt.",
        ).grid(row=1, column=0, columnspan=3, sticky="ew", padx=18, pady=(0, 10))

        selection = self._card(tab)
        selection.grid(row=2, column=0, columnspan=3, sticky="ew", padx=18, pady=(0, 9))
        selection.grid_columnconfigure(1, weight=1)
        self.test_selection_labels: dict[str, ctk.CTkLabel] = {}
        self.test_selection_buttons: dict[str, ctk.CTkButton] = {}
        self._selection_row(selection, 0, "Testdatensatz", "Klassenordner wählen …")
        self._selection_row(selection, 1, "Keras-Modell", ".keras-Modell wählen …")
        self.test_selection_status = ctk.CTkLabel(
            selection,
            text="Noch kein Testdatensatz geladen",
            text_color=TEAL,
            font=ctk.CTkFont(family=self.font_family, size=13, weight="bold"),
            anchor="w",
        )
        self.test_selection_status.grid(row=2, column=0, columnspan=3, sticky="ew", padx=15, pady=(7, 5))

        evaluation_controls = ctk.CTkFrame(selection, fg_color="transparent")
        evaluation_controls.grid(row=3, column=0, columnspan=3, sticky="ew", padx=14, pady=(2, 12))
        evaluation_controls.grid_columnconfigure(1, weight=1)
        self.evaluate_button = ctk.CTkButton(
            evaluation_controls,
            text="Auswertung starten",
            width=166,
            height=40,
            fg_color=TEAL,
            hover_color="#066C77",
            font=ctk.CTkFont(family=self.font_family, size=14, weight="bold"),
            command=self._start_evaluation,
            state="disabled",
        )
        self.evaluate_button.grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.evaluation_status = ctk.CTkLabel(
            evaluation_controls,
            text=" ",
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=13),
            anchor="w",
        )
        self.evaluation_status.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        self.evaluation_progress = ctk.CTkProgressBar(
            evaluation_controls, width=150, height=10, progress_color=TEAL, fg_color="#E7EDF0"
        )
        self.evaluation_progress.set(0)
        self.evaluation_progress.grid(row=0, column=2, sticky="e")
        self._evaluation_controls = [
            *self.test_selection_buttons.values(),
            self.evaluate_button,
        ]

        self.metric_value_labels: dict[str, ctk.CTkLabel] = {}

        table_card = self._card(tab)
        table_card.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=18, pady=(0, 9))
        table_card.grid_columnconfigure(0, weight=1)
        table_title_row = ctk.CTkFrame(table_card, fg_color="transparent")
        table_title_row.grid(row=0, column=0, sticky="ew", padx=13, pady=(9, 5))
        ctk.CTkLabel(
            table_title_row,
            text="Ergebnisse pro Testbild",
            text_color=TEAL,
            font=ctk.CTkFont(family=self.font_family, size=14, weight="bold"),
        ).pack(side="left")
        self.results_table_status = ctk.CTkLabel(
            table_title_row,
            text="Noch keine Auswertung",
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=13),
        )
        self.results_table_status.pack(side="right")

        table_frame = tk.Frame(table_card, background=CARD, highlightthickness=0)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=11, pady=(0, 10))
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)
        columns = ("true", "predicted", "broken", "intact", "result")
        self.results_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="tree headings",
            style="Prototype.Treeview",
            selectmode="browse",
            height=14,
        )
        self.results_table.heading("#0", text="Testbild")
        self.results_table.column("#0", width=260, minwidth=145, stretch=True)
        headings = {
            "true": "Wahres Label",
            "predicted": "Vorhersage",
            "broken": "% (Gebrochen)",
            "intact": "% (Intakt)",
            "result": "Ergebnis",
        }
        for column, heading in headings.items():
            self.results_table.heading(column, text=heading)
            self.results_table.column(column, width=135, minwidth=90, anchor="center", stretch=True)
        self.results_table.tag_configure("correct", background=GREEN_SOFT, foreground="#255F3B")
        self.results_table.tag_configure("incorrect", background=RED_SOFT, foreground="#8B3535")
        self.results_table.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_table.yview)
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.results_table.xview)
        self.results_table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")

        evaluation = self._card(tab)
        evaluation.grid(row=5, column=0, columnspan=3, sticky="ew", padx=18, pady=(0, 14))
        evaluation.grid_columnconfigure(0, weight=3, uniform="evaluation_result")
        evaluation.grid_columnconfigure(1, weight=2, uniform="evaluation_result")
        self._card_heading(evaluation, "Ergebnisse der Auswertung", TEAL).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=13, pady=(10, 5)
        )

        token_row = ctk.CTkFrame(evaluation, fg_color="transparent")
        token_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
        token_row.grid_columnconfigure(0, weight=1)
        self.student_comparison_entry = ctk.CTkEntry(
            token_row,
            placeholder_text="Token eingeben",
            show="*",
            height=36,
            font=ctk.CTkFont(family=self.font_family, size=13),
        )
        self.student_comparison_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            token_row,
            text="Ergebnisse vergleichen",
            width=176,
            height=36,
            fg_color=TEAL,
            hover_color="#066C77",
            font=ctk.CTkFont(family=self.font_family, size=13, weight="bold"),
            command=self._compare_student_results,
        ).grid(row=0, column=1, sticky="e")

        self._card_heading(evaluation, "Konfusionsmatrix", TEAL).grid(
            row=2, column=0, sticky="w", padx=13, pady=(2, 3)
        )
        matrix_frame = ctk.CTkFrame(evaluation, fg_color="transparent")
        matrix_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        matrix_frame.grid_columnconfigure(0, minsize=100)
        matrix_frame.grid_columnconfigure((1, 2), weight=1, uniform="matrix_cells")
        ctk.CTkLabel(
            matrix_frame,
            text="Tats\u00e4chlich / Vorhersage",
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=12, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(3, 5), pady=(0, 3))
        for column, class_name in enumerate(("Gebrochen", "Intakt"), start=1):
            ctk.CTkLabel(
                matrix_frame,
                text=class_name,
                text_color=TEAL,
                font=ctk.CTkFont(family=self.font_family, size=12, weight="bold"),
                anchor="center",
                justify="center",
                wraplength=120,
            ).grid(row=0, column=column, sticky="ew", padx=3, pady=(0, 3))

        self.confusion_matrix_values: dict[tuple[int, int], ctk.CTkLabel] = {}
        for true_index, true_name in enumerate(("Gebrochen", "Intakt")):
            row = true_index + 1
            ctk.CTkLabel(
                matrix_frame,
                text=true_name,
                text_color=TEXT,
                font=ctk.CTkFont(family=self.font_family, size=12, weight="bold"),
                anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=(3, 5), pady=2)
            for predicted_index in range(2):
                correct = true_index == predicted_index
                cell = ctk.CTkLabel(
                    matrix_frame,
                    text="--",
                    fg_color=GREEN_SOFT if correct else RED_SOFT,
                    text_color="#255F3B" if correct else "#8B3535",
                    corner_radius=6,
                    height=30,
                    font=ctk.CTkFont(family=self.font_family, size=17, weight="bold"),
                    anchor="center",
                )
                cell.grid(row=row, column=predicted_index + 1, sticky="ew", padx=3, pady=2)
                self.confusion_matrix_values[(true_index, predicted_index)] = cell

        count_cards = ctk.CTkFrame(evaluation, fg_color="transparent")
        count_cards.grid(row=2, column=1, rowspan=2, sticky="nsew", padx=(8, 13), pady=(2, 6))
        count_cards.grid_columnconfigure((0, 1), weight=1, uniform="classification_counts")
        count_cards.grid_rowconfigure((0, 1), weight=1, uniform="classification_count_rows")
        self.reference_count_labels: dict[str, ctk.CTkLabel] = {}
        self.rate_value_labels: dict[str, ctk.CTkLabel] = {}
        count_specs = (
            ("fp", "Falsch-positiv (fp)", None),
            ("fn", "Falsch-negativ (fn)", None),
            ("tp", "Richtig-positiv (rp)", None),
            ("tn", "Richtig-negativ (rn)", None),
        )
        for index, (key, title, rate_key) in enumerate(count_specs):
            row, column = divmod(index, 2)
            count_card = ctk.CTkFrame(count_cards, fg_color="#EAF0F3", corner_radius=8)
            count_card.grid(row=row, column=column, sticky="nsew", padx=3, pady=3)
            count_card.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                count_card,
                text=title,
                text_color=MUTED,
                font=ctk.CTkFont(family=self.font_family, size=11),
                anchor="center",
                justify="center",
                wraplength=150,
            ).grid(row=0, column=0, sticky="ew", padx=5, pady=(4, 0))
            count_value = ctk.CTkLabel(
                count_card,
                text="--",
                text_color=TEAL,
                font=ctk.CTkFont(family=self.font_family, size=17, weight="bold"),
                anchor="center",
            )
            count_value.grid(row=1, column=0, sticky="ew", padx=5)
            self.reference_count_labels[key] = count_value
            rate_label = ctk.CTkLabel(
                count_card,
                text="",
                text_color=MUTED,
                font=ctk.CTkFont(family=self.font_family, size=10),
                anchor="center",
            )
            rate_label.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 4))
            if rate_key is not None:
                self.rate_value_labels[rate_key] = rate_label

        metrics_row = ctk.CTkFrame(evaluation, fg_color="transparent")
        metrics_row.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        metrics_row.grid_columnconfigure(tuple(range(5)), weight=1, uniform="evaluation_metric_cards")
        metric_specs = (
            ("accuracy", "Genauigkeit"),
            ("sensitivity", "Sensitivit\u00e4t"),
            ("specificity", "Spezifit\u00e4t"),
            ("fpr", "False Positive Rate (FPR)"),
            ("fnr", "False Negative Rate (FNR)"),
        )
        metric_cards = []
        for column, (key, title) in enumerate(metric_specs):
            metric = self._card(metrics_row)
            metric_cards.append(metric)
            metric.grid(row=0, column=column, sticky="ew", padx=4, pady=2)
            ctk.CTkLabel(
                metric,
                text=title,
                text_color=MUTED,
                font=ctk.CTkFont(family=self.font_family, size=13),
            ).pack(anchor="w", padx=13, pady=(9, 0))
            value_label = ctk.CTkLabel(
                metric,
                text="--",
                text_color=TEAL,
                font=ctk.CTkFont(family=self.font_family, size=20, weight="bold"),
            )
            value_label.pack(anchor="w", padx=13, pady=(0, 9))
            self.metric_value_labels[key] = value_label

        def arrange_metric_cards(event: tk.Event) -> None:
            columns = 5 if event.width >= 850 else 3
            for column in range(5):
                metrics_row.grid_columnconfigure(column, weight=1, uniform="evaluation_metric_cards")
            for index, metric in enumerate(metric_cards):
                row, column = divmod(index, columns)
                metric.grid(row=row, column=column, sticky="ew", padx=4, pady=2)

        metrics_row.bind("<Configure>", arrange_metric_cards)

    def _selection_row(self, parent: ctk.CTkFrame, row: int, label: str, button_text: str) -> None:
        ctk.CTkLabel(
            parent,
            text=label,
            width=112,
            text_color=TEXT,
            font=ctk.CTkFont(family=self.font_family, size=13, weight="bold"),
            anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(14, 4), pady=5)
        path_label = ctk.CTkLabel(
            parent,
            text="Noch nicht ausgewählt",
            text_color=MUTED,
            font=ctk.CTkFont(family=self.font_family, size=13),
            anchor="w",
        )
        path_label.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.test_selection_labels[label] = path_label
        action = ctk.CTkFrame(parent, fg_color="transparent")
        action.grid(row=row, column=2, sticky="e", padx=(4, 14), pady=5)
        icon = tk.Canvas(action, width=20, height=20, background=CARD, highlightthickness=0)
        icon.pack(side="left", padx=(0, 4))
        self._draw_icon(icon, "folder" if row == 0 else "export", TEAL)
        button = ctk.CTkButton(
            action,
            text=button_text,
            width=156,
            height=40,
            fg_color="#EDF2F5",
            hover_color="#DDE7EC",
            text_color=TEAL,
            font=ctk.CTkFont(family=self.font_family, size=14),
            command=self._select_test_folder if row == 0 else self._select_test_model,
        )
        button.pack(side="left")
        self.test_selection_buttons[label] = button

    def _select_test_folder(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.root,
            title="Unabhängigen Testordner auswählen",
            mustexist=True,
        )
        if not selected:
            return
        folder = Path(selected).expanduser().resolve()
        try:
            if not folder.is_dir():
                raise ValueError("Der ausgewählte Testordner ist nicht verfuegbar.")
            self._validate_test_folder_separation(folder)
            labelled_paths = self._discover_labeled_test_images(folder)
            self._validate_test_file_separation(labelled_paths)
        except (OSError, ValueError) as error:
            self.test_folder = None
            self.test_records.clear()
            self.test_selection_labels["Testdatensatz"].configure(
                text="Kein gültiger Testdatensatz ausgewählt", text_color=RED
            )
            self.test_selection_status.configure(text=str(error), text_color=RED)
            self._clear_evaluation_display()
            self._update_evaluation_availability()
            messagebox.showerror("Ungültiger Testdatensatz", str(error), parent=self.root)
            return

        self.test_folder = folder
        self.test_records = [
            {"path": path, "true_label": label, "predicted_label": None, "probabilities": None}
            for path, label in labelled_paths
        ]
        self.test_selection_labels["Testdatensatz"].configure(text=str(folder), text_color=TEXT)
        counts = [sum(label == index for _path, label in labelled_paths) for index in range(2)]
        self.test_selection_status.configure(
            text=f"{len(labelled_paths)} Bilder geladen | Gebrochen: {counts[0]} | Intakt: {counts[1]}",
            text_color=TEAL,
        )
        self._clear_evaluation_display()
        self.evaluation_status.configure(text="Testbilder geladen. Modell auswählen.", text_color=MUTED)
        self._update_evaluation_availability()

    @staticmethod
    def _discover_labeled_test_images(folder: Path) -> list[tuple[Path, int]]:
        """Recursively discover supported test images and label by class subfolder."""
        if not folder.is_dir():
            raise ValueError(f"Testordner nicht gefunden: {folder}")
        labelled_paths: list[tuple[Path, int]] = []
        for label, class_name in enumerate(CLASS_NAMES):
            class_folder = folder / class_name
            if not class_folder.is_dir():
                raise ValueError(f"Der erforderliche Klassenordner fehlt: {class_folder}")
            class_paths = sorted(
                path for path in class_folder.rglob("*")
                if path.is_file() and path.suffix.lower() in TEST_IMAGE_EXTENSIONS
            )
            if not class_paths:
                raise ValueError(
                    f"Keine JPG-, JPEG- oder PNG-Bilder im Klassenordner gefunden: {class_folder}"
                )
            labelled_paths.extend((path, label) for path in class_paths)
        return labelled_paths

    @staticmethod
    def _folders_overlap(first: Path, second: Path) -> bool:
        first_resolved = first.resolve()
        second_resolved = second.resolve()
        return (
            first_resolved == second_resolved
            or first_resolved in second_resolved.parents
            or second_resolved in first_resolved.parents
        )

    @staticmethod
    def _training_image_paths(folder: Path) -> set[Path]:
        return {
            path.resolve()
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS
        }

    def _validate_test_folder_separation(self, test_folder: Path) -> None:
        for class_name in CLASS_NAMES:
            training_folder = self.training_folders.get(class_name)
            if training_folder is not None and self._folders_overlap(test_folder, training_folder):
                raise ValueError(
                    f"Der Testordner ueberlappt mit dem Trainingsordner fuer {class_name}. "
                    "Trainings- und Testdaten muessen getrennt bleiben."
                )

    def _validate_test_file_separation(self, labelled_paths: list[tuple[Path, int]]) -> None:
        training_paths: set[Path] = set()
        for training_folder in self.training_folders.values():
            if training_folder is not None:
                training_paths.update(self._training_image_paths(training_folder))
        test_paths = {path.resolve() for path, _label in labelled_paths}
        if training_paths & test_paths:
            raise ValueError(
                "Mindestens ein Testbild ist auch im ausgewählten Trainingsdatensatz enthalten. "
                "Die Auswertung wurde abgebrochen."
            )

    def _select_test_model(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Zu evaluierendes Keras-Modell auswählen",
            filetypes=(("Keras-Modell", "*.keras"),),
        )
        if not selected:
            return
        model_path = Path(selected).expanduser().resolve()
        if model_path.suffix.lower() != ".keras" or not model_path.is_file():
            self.test_model_path = None
            self.test_selection_labels["Keras-Modell"].configure(
                text="Kein gueltiges Modell ausgewählt", text_color=RED
            )
            self.evaluation_status.configure(text="Bitte wählen Sie eine vorhandene .keras-Datei.", text_color=RED)
            self._clear_evaluation_display()
            self._update_evaluation_availability()
            messagebox.showerror(
                "Ungueltiges Modell", "Bitte wählen Sie eine vorhandene .keras-Datei aus.", parent=self.root
            )
            return
        self.test_model_path = model_path
        self.test_selection_labels["Keras-Modell"].configure(text=str(model_path), text_color=TEXT)
        self._clear_evaluation_display()
        self.evaluation_status.configure(text="Modell ausgewählt. Auswertung bereit.", text_color=MUTED)
        self._update_evaluation_availability()

    def _update_evaluation_availability(self) -> None:
        ready = (
            self.test_folder is not None
            and bool(self.test_records)
            and self.test_model_path is not None
            and self.test_model_path.is_file()
            and not self._training_active
            and not self._evaluation_active
        )
        self.evaluate_button.configure(state="normal" if ready else "disabled")
        for button in self.test_selection_buttons.values():
            button.configure(state="disabled" if self._training_active or self._evaluation_active else "normal")
        self.train_button.configure(state="disabled" if self._training_active or self._evaluation_active else "normal")

    def _start_evaluation(self) -> None:
        if self._evaluation_active:
            return
        if self._training_active:
            messagebox.showinfo(
                "Training läuft",
                "Bitte warten Sie, bis das Training abgeschlossen ist, bevor Sie eine Auswertung starten.",
                parent=self.root,
            )
            return
        if self.test_folder is None:
            messagebox.showerror(
                "Testdatensatz fehlt", "Wählen Sie einen Testordner mit gebrochen/ und intakt/ aus.", parent=self.root
            )
            return
        if self.test_model_path is None or not self.test_model_path.is_file() or self.test_model_path.suffix.lower() != ".keras":
            messagebox.showerror(
                "Modell fehlt", "Wählen Sie eine vorhandene .keras-Modell-Datei aus.", parent=self.root
            )
            return
        if not self.test_records:
            messagebox.showerror("Keine Testbilder", "Der Testdatensatz enthält keine gueltigen Bilder.", parent=self.root)
            return
        try:
            self._validate_test_folder_separation(self.test_folder)
            labelled_paths = [(Path(record["path"]), int(record["true_label"])) for record in self.test_records]
            self._validate_test_file_separation(labelled_paths)
        except (OSError, ValueError) as error:
            self._clear_evaluation_display()
            self.evaluation_status.configure(text=str(error), text_color=RED)
            messagebox.showerror("Trainings- und Testdaten pruefen", str(error), parent=self.root)
            return

        self._clear_evaluation_display()
        self._evaluation_active = True
        self._update_evaluation_availability()
        self.evaluation_progress.set(0)
        self.evaluation_status.configure(text=f"Auswertung läuft: 0/{len(labelled_paths)} Bilder", text_color=AMBER)
        worker = threading.Thread(
            target=self._evaluation_worker,
            args=(self.test_model_path, labelled_paths),
            name="quality-model-evaluation",
            daemon=True,
        )
        worker.start()
        self._evaluation_poll_id = self.root.after(100, self._poll_evaluation_queue)

    def _evaluation_worker(self, model_path: Path, test_records: list[tuple[Path, int]]) -> None:
        try:
            import tensorflow as tf

            try:
                from evaluate import calculate_metrics
                from preprocessing import load_image_for_model
            except ModuleNotFoundError as import_error:
                if import_error.name not in {"evaluate", "preprocessing"}:
                    raise
                from .evaluate import calculate_metrics
                from .preprocessing import load_image_for_model

            model = tf.keras.models.load_model(model_path)
            input_shape = model.input_shape
            if (
                not isinstance(input_shape, tuple)
                or len(input_shape) != 4
                or any(actual not in (None, expected) for actual, expected in zip(input_shape[1:], (180, 180, 3)))
            ):
                raise ValueError("Das Modell muss Eingabebilder mit 180 x 180 Pixeln und 3 Farbkanälen erwarten.")
            output_shape = model.output_shape
            if not isinstance(output_shape, tuple) or not output_shape or output_shape[-1] != len(CLASS_NAMES):
                raise ValueError("Das Modell muss zwei Ausgabewerte in der Reihenfolge gebrochen, intakt liefern.")

            true_labels: list[int] = []
            predicted_labels: list[int] = []
            for index, (image_path, true_label) in enumerate(test_records, start=1):
                try:
                    image_batch = load_image_for_model(image_path)
                    probabilities = model.predict(image_batch, verbose=0)[0]
                    values = [float(value) for value in probabilities]
                    if len(values) != len(CLASS_NAMES) or not all(math.isfinite(value) for value in values):
                        raise ValueError("Das Modell lieferte keine zwei endlichen Wahrscheinlichkeiten.")
                    if any(value < 0.0 or value > 1.0 for value in values):
                        raise ValueError("Die Modellwahrscheinlichkeiten muessen zwischen 0 und 1 liegen.")
                    if not math.isclose(sum(values), 1.0, rel_tol=1e-3, abs_tol=1e-3):
                        raise ValueError("Die Modellwahrscheinlichkeiten muessen zusammen 1 ergeben.")
                except Exception as error:
                    raise ValueError(f"Bild {image_path.name} konnte nicht ausgewertet werden: {error}") from error

                predicted_label = max(range(len(values)), key=values.__getitem__)
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
                self._evaluation_queue.put((
                    "prediction",
                    {
                        "index": index - 1,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "probabilities": values,
                    },
                ))
                self._evaluation_queue.put(("progress", (index, len(test_records))))

            self._evaluation_queue.put(("complete", calculate_metrics(true_labels, predicted_labels)))
        except Exception as error:
            self._evaluation_queue.put(("error", str(error)))

    def _poll_evaluation_queue(self) -> None:
        processed = 0
        try:
            while processed < 50:
                event, payload = self._evaluation_queue.get_nowait()
                processed += 1
                if event == "prediction" and isinstance(payload, dict):
                    self._display_test_prediction(payload)
                elif event == "progress" and isinstance(payload, tuple):
                    completed, total = payload
                    self.evaluation_progress.set(completed / max(1, total))
                    self.evaluation_status.configure(
                        text=f"Auswertung läuft: {completed}/{total} Bilder", text_color=AMBER
                    )
                elif event == "complete" and isinstance(payload, dict):
                    self._handle_evaluation_complete(payload)
                elif event == "error":
                    self._handle_evaluation_error(str(payload))
        except queue.Empty:
            pass
        if self._evaluation_active:
            try:
                self._evaluation_poll_id = self.root.after(100, self._poll_evaluation_queue)
            except tk.TclError:
                self._evaluation_poll_id = None

    def _display_test_prediction(self, result: dict[str, object]) -> None:
        index = int(result["index"])
        true_label = int(result["true_label"])
        predicted_label = int(result["predicted_label"])
        probabilities = [float(value) for value in result["probabilities"]]
        record = self.test_records[index]
        record["predicted_label"] = predicted_label
        record["probabilities"] = probabilities
        correct = predicted_label == true_label
        self.results_table.insert(
            "", "end",
            text=Path(record["path"]).relative_to(self.test_folder).as_posix(),
            values=(
                CLASS_NAMES[true_label].capitalize(),
                f"{CLASS_NAMES[predicted_label].capitalize()} {'(korrekt)' if correct else '(abweichend)'}",
                f"{probabilities[0] * 100:.2f}%",
                f"{probabilities[1] * 100:.2f}%",
                "Korrekt" if correct else "Abweichung",
            ),
            tags=("correct" if correct else "incorrect",),
        )

    def _handle_evaluation_complete(self, metrics: dict[str, object]) -> None:
        labels = [int(record["true_label"]) for record in self.test_records]
        missing = []
        if 0 not in labels:
            missing.append("Gebrochen fehlt; Sensitivität/FNR sind nicht definiert")
        if 1 not in labels:
            missing.append("Intakt fehlt; Spezifität/FPR sind nicht definiert")
        self.completed_evaluation = {
            "model_path": self.test_model_path,
            "test_folder": self.test_folder,
            "evaluated_at": datetime.now().astimezone(),
            "records": [dict(record) for record in self.test_records],
            "metrics": metrics,
        }
        self._refresh_evaluation_export_buttons()
        self._render_protected_metrics()
        self.results_table_status.configure(text=f"Auswertung abgeschlossen: {len(labels)} Testbilder", text_color=GREEN)
        self.evaluation_status.configure(
            text="Auswertung abgeschlossen" + (" | " + "; ".join(missing) if missing else ""),
            text_color=AMBER if missing else GREEN,
        )
        self.evaluation_progress.set(1)
        self._finish_evaluation()

    @staticmethod
    def _format_evaluation_metric(value: object) -> str:
        return "N/A" if value is None else f"{float(value) * 100:.2f}%"

    def _handle_evaluation_error(self, error: str) -> None:
        self._clear_evaluation_display()
        self.evaluation_status.configure(text="Auswertung fehlgeschlagen: " + error, text_color=RED)
        self.results_table_status.configure(text="Auswertung fehlgeschlagen", text_color=RED)
        self.evaluation_progress.set(0)
        self._finish_evaluation()
        messagebox.showerror("Auswertung fehlgeschlagen", error, parent=self.root)

    def _clear_evaluation_display(self) -> None:
        self.completed_evaluation = None
        self._comparison_authorized = False
        self._refresh_evaluation_export_buttons()
        for item in self.results_table.get_children():
            self.results_table.delete(item)
        for label in self.metric_value_labels.values():
            label.configure(text="--")
        for label in self.confusion_matrix_values.values():
            label.configure(text="--")
        for label in self.rate_value_labels.values():
            label.configure(text="")
        for label in self.reference_count_labels.values():
            label.configure(text="--")
        self.results_table_status.configure(text="Noch keine Auswertung", text_color=MUTED)
        self.evaluation_progress.set(0)

    def _finish_evaluation(self) -> None:
        self._evaluation_active = False
        self._set_training_controls_enabled(True)
        self._update_evaluation_availability()
        self._refresh_evaluation_export_buttons()

    def _build_export_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=1)
        self._heading(tab, "Export", PURPLE, "export").grid(row=0, column=0, sticky="w", padx=18, pady=(18, 3))
        self._description(
            tab,
            "CSV- und TXT-Berichte sind nach erfolgreicher Auswertung verfügbar. ",
        ).grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 14))

        sections = (
            ("Trainiertes Modell", "Ein trainiertes Modell als eigenständige .keras-Datei speichern."),
            ("Auswertung als CSV", "Die letzte erfolgreich abgeschlossene Auswertung als CSV-Datei speichern."),
            ("Bericht als TXT", "Die letzte erfolgreich abgeschlossene Auswertung als Textbericht speichern."),
        )
        for row, (title, description) in enumerate(sections, start=2):
            card = self._card(tab)
            card.grid(row=row, column=0, sticky="ew", padx=18, pady=6)
            card.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                card,
                text=title,
                text_color=PURPLE,
                font=ctk.CTkFont(family=self.font_family, size=17, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=15, pady=(12, 2))
            description_label = ctk.CTkLabel(
                card,
                text=description,
                text_color=MUTED,
                font=ctk.CTkFont(family=self.font_family, size=13),
                anchor="w",
                justify="left",
            )
            description_label.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 12))
            self._bind_wraplength(description_label, card, padding=190)
            action = "Speicherort wählen" if row == 2 else "Exportziel wählen"
            command = self._export_placeholder
            state = "normal"
            if row == 3:
                command = lambda: self._export_evaluation_results(".csv")
                state = "disabled"
            elif row == 4:
                command = lambda: self._export_evaluation_results(".txt")
                state = "disabled"
            button = ctk.CTkButton(
                card,
                text=action,
                width=190,
                height=40,
                fg_color="#F0ECFF",
                hover_color="#E2DBFF",
                text_color=PURPLE,
                font=ctk.CTkFont(family=self.font_family, size=14),
                command=command,
                state=state,
            )
            button.grid(row=0, column=1, rowspan=2, padx=15, pady=12)
            if row == 3:
                self.csv_export_button = button
            elif row == 4:
                self.txt_export_button = button
        self._refresh_evaluation_export_buttons()

    def _refresh_evaluation_export_buttons(self) -> None:
        available = (
            self._has_reference_access()
            and self.completed_evaluation is not None
            and not self._evaluation_active
        )
        for button in (getattr(self, "csv_export_button", None), getattr(self, "txt_export_button", None)):
            if button is not None:
                button.configure(state="normal" if available else "disabled")

    def _export_evaluation_results(self, extension: str) -> None:
        if not self._has_reference_access():
            self._refresh_evaluation_export_buttons()
            messagebox.showerror(
                "Token erforderlich",
                "Bitte geben Sie den gültigen Sitzungstoken ein, um Auswertungsberichte zu exportieren.",
                parent=self.root,
            )
            return
        result = self.completed_evaluation
        if result is None or self._evaluation_active:
            self._refresh_evaluation_export_buttons()
            messagebox.showinfo(
                "Keine abgeschlossene Auswertung",
                "Führen Sie zuerst eine Auswertung erfolgreich durch.",
                parent=self.root,
            )
            return
        if extension not in (".csv", ".txt"):
            messagebox.showerror("Ungültiges Exportformat", "Nur CSV- und TXT-Berichte werden unterstützt.", parent=self.root)
            return

        evaluated_at = result["evaluated_at"]
        timestamp = evaluated_at.strftime("%Y%m%d_%H%M%S")
        results_dir = self.project_root / "results"
        initial_dir = results_dir if results_dir.is_dir() else self.project_root
        filetypes = (("CSV-Datei", "*.csv"),) if extension == ".csv" else (("Textbericht", "*.txt"),)
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Auswertung als CSV speichern" if extension == ".csv" else "Auswertung als TXT speichern",
            initialdir=str(initial_dir),
            initialfile=f"auswertung_{timestamp}{extension}",
            defaultextension=extension,
            filetypes=filetypes,
            confirmoverwrite=False,
        )
        if not selected:
            return

        destination = Path(selected).expanduser()
        if not destination.suffix:
            destination = destination.with_suffix(extension)
        if destination.suffix.lower() != extension:
            messagebox.showerror(
                "Dateiendung passt nicht",
                f"Bitte wählen Sie eine Datei mit der Endung {extension}.",
                parent=self.root,
            )
            return
        try:
            destination = destination.resolve()
            if destination.is_dir() or not destination.parent.is_dir():
                raise ValueError("Der Zielordner ist ungültig oder existiert nicht.")
        except (OSError, ValueError) as error:
            messagebox.showerror("Ungültiger Exportpfad", str(error), parent=self.root)
            return
        if destination.exists() and not messagebox.askyesno(
            "Datei überschreiben?",
            f"Die Datei existiert bereits:\n{destination}\n\nSoll sie überschrieben werden?",
            parent=self.root,
        ):
            return

        try:
            try:
                from evaluation_export import write_evaluation_csv, write_evaluation_text
            except ModuleNotFoundError as import_error:
                if import_error.name != "evaluation_export":
                    raise
                from .evaluation_export import write_evaluation_csv, write_evaluation_text
            if extension == ".csv":
                write_evaluation_csv(destination, result)
            else:
                write_evaluation_text(destination, result)
        except Exception as error:
            messagebox.showerror(
                "Export fehlgeschlagen",
                f"Der Bericht konnte nicht gespeichert werden:\n{error}",
                parent=self.root,
            )
            return
        messagebox.showinfo(
            "Export erfolgreich",
            f"Der Bericht wurde gespeichert unter:\n{destination}",
            parent=self.root,
        )

    def _select_training_folder(self, class_name: str) -> None:
        folder = filedialog.askdirectory(
            title=f"Ordner für {class_name} auswählen",
            parent=self.root,
            mustexist=True,
        )
        if not folder:
            return
        path = Path(folder).expanduser().resolve()
        self.training_folders[class_name] = path
        self.training_folder_labels[class_name].configure(text=str(path), text_color=TEXT)
        self._refresh_training_summary()
        self.training_action_status.configure(
            text="Trainingsordner ausgewählt; Ausgabepfad prüfen.", text_color=MUTED
        )

    def _refresh_training_summary(self) -> None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        counts: dict[str, int | None] = {}
        for class_name, folder in self.training_folders.items():
            if folder is None or not folder.is_dir():
                counts[class_name] = None
            else:
                counts[class_name] = sum(
                    item.is_file() and item.suffix.lower() in extensions
                    for item in folder.iterdir()
                )
        broken_count = "?" if counts["gebrochen"] is None else str(counts["gebrochen"])
        intact_count = "?" if counts["intakt"] is None else str(counts["intakt"])
        self.training_summary.configure(
            text=(
                f"Bildanzahl: Gebrochen {broken_count} | Intakt {intact_count}"
                "   |   Geplante Aufteilung: 80 % Training / 20 % Validierung je Klasse"
            )
        )

    def _select_model_output(self) -> None:
        initial_dir = self.project_root / "models"
        if not initial_dir.is_dir():
            initial_dir = self.project_root
        selected = filedialog.asksaveasfilename(
            title="Speicherort für das trainierte Modell auswählen",
            parent=self.root,
            initialdir=str(initial_dir),
            initialfile="quality_model_gui.keras",
            defaultextension=".keras",
            filetypes=(("Keras-Modell", "*.keras"),),
        )
        if not selected:
            return
        output_path = Path(selected).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(".keras")
        if output_path.suffix.lower() != ".keras":
            messagebox.showerror(
                "Ungültige Dateiendung",
                "Das Modell muss als .keras-Datei gespeichert werden.",
                parent=self.root,
            )
            return
        output_path = output_path.resolve()
        if output_path.exists():
            messagebox.showwarning(
                "Datei existiert bereits",
                "Das vorhandene Modell wird nicht überschrieben. Bitte wählen Sie einen neuen Dateinamen.",
                parent=self.root,
            )
            return
        self._selected_model_output = output_path
        self.model_path_status.configure(text=f"Modellziel: {output_path}", text_color=TEXT)
        self.training_action_status.configure(text="Ausgabepfad ausgewählt.", text_color=NAVY)

    def _validate_training_form(self) -> tuple[Path, Path, int, int, float, Path]:
        broken_dir = self.training_folders["gebrochen"]
        intact_dir = self.training_folders["intakt"]
        if broken_dir is None or intact_dir is None:
            raise ValueError("Bitte wählen Sie je einen Ordner für Gebrochen und Intakt aus.")
        for class_name, folder in (("Gebrochen", broken_dir), ("Intakt", intact_dir)):
            if not folder.is_dir():
                raise ValueError(f"Der ausgewählte Ordner für {class_name} ist nicht mehr verfügbar.")
            image_count = sum(
                path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
                for path in folder.iterdir()
            )
            if image_count < 2:
                raise ValueError(
                    f"Der Ordner für {class_name} braucht mindestens zwei unterstützte Bilder "
                    f"(gefunden: {image_count})."
                )
        if broken_dir.resolve() == intact_dir.resolve():
            raise ValueError("Bitte wählen Sie zwei verschiedene Klassenordner aus.")

        independent_test_root = (self.project_root / "data" / "test").resolve()
        for folder in (broken_dir.resolve(), intact_dir.resolve()):
            if folder == independent_test_root or folder in independent_test_root.parents or independent_test_root in folder.parents:
                raise ValueError(
                    "Der Ordner data/test ist für die unabhängige Prüfung reserviert und darf nicht "
                    "für Training oder Validierung verwendet werden."
                )

        if self.test_folder is not None:
            selected_training_folders = (broken_dir.resolve(), intact_dir.resolve())
            if any(self._folders_overlap(self.test_folder, folder) for folder in selected_training_folders):
                raise ValueError(
                    "Der ausgewählte Testdatensatz darf nicht fuer Training oder Validierung verwendet werden."
                )
            test_files = {Path(record["path"]).resolve() for record in self.test_records}
            training_files: set[Path] = set()
            for folder in selected_training_folders:
                training_files.update(self._training_image_paths(folder))
            if test_files & training_files:
                raise ValueError(
                    "Mindestens ein Testbild ist auch im Trainingsdatensatz enthalten. Training wurde nicht gestartet."
                )

        try:
            epochs = int(self.training_entries["Epochen"].get().strip())
            batch_size = int(self.training_entries["Batchgröße"].get().strip())
            learning_rate = float(self.training_entries["Lernrate"].get().strip())
        except ValueError as error:
            raise ValueError("Epochen und Batchgröße müssen ganze Zahlen und die Lernrate eine Zahl sein.") from error
        if epochs <= 0:
            raise ValueError("Epochen muss eine positive ganze Zahl sein.")
        if batch_size <= 0:
            raise ValueError("Die Batchgröße muss eine positive ganze Zahl sein.")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Die Lernrate muss eine endliche Zahl größer als 0 sein.")

        output_path = self._selected_model_output
        if output_path is None:
            raise ValueError("Bitte wählen Sie zuerst einen Speicherort für das .keras-Modell.")
        if output_path.suffix.lower() != ".keras":
            raise ValueError("Der Modellpfad muss auf .keras enden.")
        if not output_path.parent.is_dir():
            raise ValueError(f"Der Zielordner existiert nicht: {output_path.parent}")
        if output_path.exists():
            raise FileExistsError(
                f"Die Datei existiert bereits und wird nicht überschrieben: {output_path}. "
                "Bitte wählen Sie einen anderen Namen."
            )
        return broken_dir, intact_dir, epochs, batch_size, learning_rate, output_path

    def _start_training(self) -> None:
        if self._training_active:
            return
        if self._evaluation_active:
            messagebox.showinfo(
                "Auswertung läuft",
                "Bitte warten Sie, bis die Auswertung abgeschlossen ist, bevor Sie ein Training starten.",
                parent=self.root,
            )
            return
        try:
            arguments = self._validate_training_form()
        except (OSError, ValueError) as error:
            messagebox.showerror("Trainingsdaten prüfen", str(error), parent=self.root)
            return

        self._training_active = True
        self._set_training_controls_enabled(False)
        self._update_evaluation_availability()
        self._training_history_epoch_count = 0
        self._set_training_history("Training wird vorbereitet...", replace=True)
        self.training_progress.set(0)
        self.training_action_status.configure(text="Training wird vorbereitet...", text_color=AMBER)
        worker = threading.Thread(
            target=self._training_worker,
            args=arguments,
            name="quality-model-training",
            daemon=True,
        )
        worker.start()

    def _training_worker(
        self,
        broken_dir: Path,
        intact_dir: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        output_path: Path,
    ) -> None:
        try:
            from train import train_model_for_gui

            result = train_model_for_gui(
                broken_dir=broken_dir,
                intact_dir=intact_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_output_path=output_path,
                progress_callback=lambda metrics: self._training_queue.put(("epoch", metrics)),
            )
            self._training_queue.put(("complete", result))
        except Exception as error:
            self._training_queue.put(("error", str(error)))

    def _poll_training_queue(self) -> None:
        try:
            while True:
                event, payload = self._training_queue.get_nowait()
                if event == "epoch" and isinstance(payload, dict):
                    self._handle_training_epoch(payload)
                elif event == "complete" and isinstance(payload, dict):
                    self._handle_training_complete(payload)
                elif event == "error":
                    self._handle_training_error(str(payload))
        except queue.Empty:
            pass
        try:
            self._training_poll_id = self.root.after(100, self._poll_training_queue)
        except tk.TclError:
            self._training_poll_id = None

    def _handle_training_epoch(self, metrics: dict[str, object]) -> None:
        epoch = int(metrics.get("epoch", 0))
        epochs = max(1, int(metrics.get("epochs", 1)))
        self.training_progress.set(min(1.0, max(0.0, epoch / epochs)))
        details = self._format_training_metrics(metrics)
        self.training_action_status.configure(
            text=f"Epoche {epoch}/{epochs} abgeschlossen{': ' + details if details else ''}",
            text_color=AMBER,
        )
        if details:
            self._append_training_history(f"Epoch {epoch}: {details}")
            self._training_history_epoch_count += 1

    def _set_training_history(self, text: str, replace: bool = False) -> None:
        self.training_history.configure(state="normal", text_color=MUTED if replace else TEXT)
        if replace:
            self.training_history.delete("1.0", "end")
        self.training_history.insert("end", text + "\n")
        self.training_history.configure(state="disabled")
        self.training_history.see("end")

    def _append_training_history(self, text: str) -> None:
        self._set_training_history(text)

    @staticmethod
    def _format_training_metrics(metrics: dict[str, object]) -> str:
        names = (
            ("loss", "Loss"),
            ("accuracy", "Accuracy"),
            ("val_loss", "Val-Loss"),
            ("val_accuracy", "Val-Accuracy"),
        )
        parts: list[str] = []
        for key, label in names:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                parts.append(f"{label} {float(value):.4f}")
        return " | ".join(parts)

    def _handle_training_complete(self, result: dict[str, object]) -> None:
        model_path = Path(result["model_path"])
        self.trained_model_path = model_path
        self.training_progress.set(1)
        epochs = result.get("completed_epochs", 0)
        training_images = result.get("training_images", 0)
        validation_images = result.get("validation_images", 0)
        self.training_action_status.configure(
            text=(
                f"Training abgeschlossen: {epochs} Epochen | "
                f"{training_images} Trainingsbilder | {validation_images} Validierungsbilder"
            ),
            text_color=GREEN,
        )
        self.model_path_status.configure(text=f"Gespeichertes Modell: {model_path}", text_color=GREEN)
        if not self._training_history_epoch_count:
            self._append_training_history("Training abgeschlossen.")
        self._finish_training()

    def _handle_training_error(self, error: str) -> None:
        self.training_action_status.configure(text=f"Training fehlgeschlagen: {error}", text_color=RED)
        messagebox.showerror("Trainingsfehler", error, parent=self.root)
        self._finish_training()

    def _finish_training(self) -> None:
        self._training_active = False
        self._set_training_controls_enabled(True)
        self._update_evaluation_availability()

    def _set_training_controls_enabled(self, enabled: bool) -> None:
        for widget in self._training_controls:
            widget.configure(
                state="normal" if enabled and not (widget is self.train_button and self._evaluation_active) else "disabled"
            )

    def _integration_placeholder(self) -> None:
        messagebox.showinfo(
            "Noch nicht verbunden",
            "Die Ordner- und Modellauswahl dient hier nur der Vorschau. "
            "Die Verbindung zur bestehenden Anwendung ist nicht implementiert.",
            parent=self.root,
        )

    @staticmethod
    def _bind_wraplength(label: ctk.CTkLabel, parent: ctk.CTkFrame, padding: int = 42) -> None:
        def resize(event: tk.Event) -> None:
            label.configure(wraplength=max(240, event.width - padding))

        parent.bind("<Configure>", resize, add="+")

    def _training_placeholder(self) -> None:
        messagebox.showinfo(
            "Visueller Prototyp",
            "Training ist in dieser Vorschau nicht implementiert. Es wird kein Modell trainiert.",
            parent=self.root,
        )

    def _export_placeholder(self) -> None:
        messagebox.showinfo(
            "Visueller Prototyp",
            "Export ist in dieser Vorschau nicht implementiert. Es werden keine Dateien erstellt.",
            parent=self.root,
        )


def main() -> None:
    try:
        application = QualityUiPrototype()
        application.run()
    except Exception as error:
        detail = f"Die Benutzeroberfläche konnte nicht gestartet werden:\n{error}"
        try:
            messagebox.showerror("Startfehler", detail)
        except tk.TclError:
            print(detail, file=sys.stderr)


if __name__ == "__main__":
    main()
