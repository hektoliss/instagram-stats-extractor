"""
Графический интерфейс для Instagram Stats Extractor
"""
import json
import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import List, Optional
import threading

from instagram_stats_extractor import extract_instagram_stats, extract_instagram_stats_from_multiple_files

# Настройка логирования для GUI
logger = logging.getLogger(__name__)

# Настраиваем логирование только если оно еще не настроено
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('instagram_gui.log', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=False
    )
else:
    # Если логирование уже настроено, добавляем только файловый handler для GUI
    if not any(isinstance(h, logging.FileHandler) and 
               hasattr(h, 'baseFilename') and 
               'instagram_gui.log' in h.baseFilename 
               for h in logger.handlers):
        file_handler = logging.FileHandler('instagram_gui.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)


class InstagramStatsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Instagram Stats Extractor")
        self.root.geometry("1200x800")
        
        # Список выбранных файлов (максимум 5)
        self.selected_files: List[str] = []
        self.results: dict = {}
        
        logger.info("Initializing GUI application")
        self.setup_ui()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        # Заголовок
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Instagram Stats Extractor",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)
        
        # Основной контейнер
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель - выбор файлов
        left_panel = tk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Правая панель - результаты
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # === Левая панель ===
        # Заголовок
        files_label = tk.Label(
            left_panel,
            text="Selected Files (up to 5)\n(processed as one post)",
            font=("Arial", 14, "bold"),
            anchor="w",
            justify=tk.LEFT
        )
        files_label.pack(fill=tk.X, pady=(0, 10))
        
        # Кнопка выбора файлов
        select_btn = tk.Button(
            left_panel,
            text="Select Files",
            command=self.select_files,
            bg="#3498db",
            fg="white",
            font=("Arial", 12),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        select_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Список файлов
        files_frame = tk.Frame(left_panel)
        files_frame.pack(fill=tk.BOTH, expand=True)
        
        self.files_listbox = tk.Listbox(
            files_frame,
            font=("Arial", 10),
            selectmode=tk.SINGLE
        )
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        files_scrollbar = tk.Scrollbar(files_frame, orient=tk.VERTICAL)
        files_scrollbar.config(command=self.files_listbox.yview)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.config(yscrollcommand=files_scrollbar.set)
        
        # Кнопка удаления выбранного файла
        remove_btn = tk.Button(
            left_panel,
            text="Remove Selected",
            command=self.remove_selected_file,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10),
            padx=10,
            pady=5,
            cursor="hand2"
        )
        remove_btn.pack(fill=tk.X, pady=(10, 10))
        
        # Кнопка очистки всех файлов
        clear_btn = tk.Button(
            left_panel,
            text="Clear All",
            command=self.clear_all_files,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 10),
            padx=10,
            pady=5,
            cursor="hand2"
        )
        clear_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Кнопка запуска анализа
        self.analyze_btn = tk.Button(
            left_panel,
            text="Start Analysis",
            command=self.start_analysis,
            bg="#27ae60",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=20,
            pady=15,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.analyze_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Прогресс бар
        self.progress = ttk.Progressbar(
            left_panel,
            mode='indeterminate'
        )
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Статус
        self.status_label = tk.Label(
            left_panel,
            text="Select files for analysis",
            font=("Arial", 10),
            fg="#7f8c8d",
            anchor="w",
            wraplength=380
        )
        self.status_label.pack(fill=tk.X)
        
        # === Правая панель ===
        # Заголовок результатов
        results_label = tk.Label(
            right_panel,
            text="Analysis Results",
            font=("Arial", 14, "bold"),
            anchor="w"
        )
        results_label.pack(fill=tk.X, pady=(0, 10))
        
        # Нотебук для вкладок с результатами
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Начальная вкладка с инструкциями
        welcome_frame = tk.Frame(self.notebook)
        self.notebook.add(welcome_frame, text="Instructions")
        
        welcome_text = scrolledtext.ScrolledText(
            welcome_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            padx=20,
            pady=20
        )
        welcome_text.pack(fill=tk.BOTH, expand=True)
        welcome_text.insert("1.0", """
Welcome to Instagram Stats Extractor!

INSTRUCTIONS:
1. Click the "Select Files" button and choose up to 5 files (videos or screenshots)
2. ALL selected files will be processed AS SCREENSHOTS OF ONE POST
3. Check the list of selected files
4. Click "Start Analysis"
5. Wait for the analysis to complete
6. Results will appear in one tab with combined statistics

SUPPORTED FORMATS:
• Video: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v
• Images: .png, .jpg, .jpeg, .bmp, .gif

IMPORTANT:
• All selected files are processed together as different screenshots of one post
• Information from all screenshots is combined to get complete statistics
• Make sure the OPENAI_API_KEY environment variable is set
• Analysis may take some time depending on the number of files
• Results can be saved via context menu (right click)
        """)
        welcome_text.config(state=tk.DISABLED)
        
    def select_files(self):
        """Select files for analysis"""
        logger.info("Opening file selection dialog")
        try:
            files = filedialog.askopenfilenames(
                title="Select files for analysis (up to 5)",
                filetypes=[
                    ("All supported", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v *.png *.jpg *.jpeg *.bmp *.gif"),
                    ("Video", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v"),
                    ("Images", "*.png *.jpg *.jpeg *.bmp *.gif"),
                    ("All files", "*.*")
                ]
            )
            
            if not files:
                logger.debug("No files selected")
                return
            
            # Limit to 5 files
            remaining_slots = 5 - len(self.selected_files)
            if remaining_slots <= 0:
                logger.warning("Attempt to add files when limit reached")
                messagebox.showwarning(
                    "Limit exceeded",
                    "You can select a maximum of 5 files. Remove some files before adding new ones."
                )
                return
            
            new_files = list(files)[:remaining_slots]
            logger.info(f"Selected files: {len(new_files)}")
            
            for file_path in new_files:
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)
                    self.files_listbox.insert(tk.END, os.path.basename(file_path))
                    logger.debug(f"Added file: {os.path.basename(file_path)}")
            
            if len(self.selected_files) >= 5:
                logger.info("Reached limit of 5 files")
                messagebox.showinfo(
                    "Limit reached",
                    f"Added {len(new_files)} file(s). Maximum of 5 files reached."
                )
            
            # Activate analysis button
            if self.selected_files:
                self.analyze_btn.config(state=tk.NORMAL)
                self.status_label.config(
                    text=f"Selected files: {len(self.selected_files)}/5 (will be processed as one post)"
                )
        except Exception as e:
            error_msg = f"Error selecting files: {str(e)}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Error", error_msg)
    
    def remove_selected_file(self):
        """Remove selected file from list"""
        selection = self.files_listbox.curselection()
        if not selection:
            messagebox.showinfo("No selection", "Select a file to remove from the list")
            return
        
        index = selection[0]
        self.files_listbox.delete(index)
        removed_file = self.selected_files.pop(index)
        
        # Удаляем вкладку с результатами, если она есть
        file_name = os.path.basename(removed_file)
        for i in range(self.notebook.index("end")):
            tab_text = self.notebook.tab(i, "text")
            if tab_text == file_name:
                self.notebook.forget(i)
                if file_name in self.results:
                    del self.results[file_name]
                break
        
            # Деактивируем кнопку, если файлов не осталось
        if not self.selected_files:
            self.analyze_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Select files for analysis")
        else:
            self.status_label.config(
                text=f"Selected files: {len(self.selected_files)}/5 (will be processed as one post)"
            )
    
    def clear_all_files(self):
        """Clear all selected files"""
        if not self.selected_files:
            return
        
        if messagebox.askyesno("Confirmation", "Remove all selected files?"):
            self.selected_files.clear()
            self.files_listbox.delete(0, tk.END)
            
            # Удаляем все вкладки с результатами, кроме инструкций
            for i in range(self.notebook.index("end") - 1, -1, -1):
                tab_text = self.notebook.tab(i, "text")
                if tab_text != "Instructions":
                    self.notebook.forget(i)
            
            self.results.clear()
            self.analyze_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Select files for analysis")
    
    def start_analysis(self):
        """Запуск анализа в отдельном потоке"""
        if not self.selected_files:
            logger.warning("Attempt to start analysis without selected files")
            messagebox.showwarning("No files", "Select files for analysis")
            return
        
        logger.info(f"Starting analysis of {len(self.selected_files)} files")
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            error_msg = "OPENAI_API_KEY environment variable is not set"
            logger.error(error_msg)
            messagebox.showerror(
                "Error",
                f"{error_msg}!\n\n"
                "Set it with:\n"
                "export OPENAI_API_KEY='sk-...'"
            )
            return
        
        # Деактивируем кнопку и запускаем прогресс
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self.status_label.config(text="Analysis started...")
        
        # Запускаем анализ в отдельном потоке
        thread = threading.Thread(target=self.analyze_files, daemon=True)
        thread.start()
        logger.debug("Analysis thread started")
    
    def analyze_files(self):
        """Analyze all files together as screenshots of one post (runs in separate thread)"""
        total_files = len(self.selected_files)
        logger.info(f"Starting analysis of {total_files} files as screenshots of one post")
        
        # Update status
        self.root.after(0, lambda: 
            self.status_label.config(
                text=f"Processing {total_files} files as one post..."
            ))
        
        try:
            # Analyze all files together
            logger.debug(f"Calling extract_instagram_stats_from_multiple_files for {total_files} files")
            stats = extract_instagram_stats_from_multiple_files(file_paths=self.selected_files)
            logger.info(f"All {total_files} files successfully analyzed together")
            
            # Create result name based on all files
            file_names = [os.path.basename(f) for f in self.selected_files]
            result_name = f"Combined Result ({len(file_names)} files)"
            
            # Save result
            self.results[result_name] = stats
            
            # Create results tab
            self.root.after(0, lambda s=stats, n=result_name: 
                self.create_results_tab(n, s))
            
        except Exception as e:
            error_msg = f"Error analyzing files: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.root.after(0, lambda msg=error_msg: 
                messagebox.showerror("Error", msg))
        
        # Complete progress
        logger.info("Analysis of all files completed")
        self.root.after(0, self.analysis_complete)
    
    def analysis_complete(self):
        """Complete analysis"""
        self.progress.stop()
        self.analyze_btn.config(state=tk.NORMAL)
        self.status_label.config(
            text=f"Analysis complete! Processed files: {len(self.results)}/{len(self.selected_files)}"
        )
    
    def create_results_tab(self, file_name: str, stats: dict):
        """Create results tab"""
        # Удаляем старую вкладку, если она существует
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == file_name:
                self.notebook.forget(i)
                break
        
        # Создаем новую вкладку
        result_frame = tk.Frame(self.notebook)
        self.notebook.add(result_frame, text=file_name)
        
        # Создаем текстовое поле с результатами
        text_widget = scrolledtext.ScrolledText(
            result_frame,
            wrap=tk.WORD,
            font=("Courier", 10),
            padx=15,
            pady=15
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Форматируем и вставляем результаты
        formatted_text = self.format_stats(stats)
        text_widget.insert("1.0", formatted_text)
        text_widget.config(state=tk.DISABLED)
        
        # Добавляем контекстное меню для сохранения
        menu = tk.Menu(text_widget, tearoff=0)
        menu.add_command(
            label="Save JSON",
            command=lambda: self.save_results(file_name, stats)
        )
        menu.add_command(
            label="Copy Text",
            command=lambda: self.copy_to_clipboard(formatted_text)
        )
        
        def show_menu(event):
            menu.post(event.x_root, event.y_root)
        
        text_widget.bind("<Button-3>", show_menu)
        
        # Переключаемся на новую вкладку
        self.notebook.select(result_frame)
    
    def format_stats(self, stats: dict) -> str:
        """Format statistics for display"""
        lines = []
        lines.append("=" * 80)
        lines.append("INSTAGRAM POST STATISTICS")
        lines.append("=" * 80)
        lines.append("")
        
        # Basic information
        lines.append("BASIC INFORMATION:")
        lines.append("-" * 80)
        lines.append(f"Post URL: {stats.get('post_url', 'no data')}")
        lines.append(f"Post Type: {stats.get('post_type', 'no data')}")
        lines.append(f"Video Duration: {stats.get('video_duration', 'no data')} sec")
        lines.append("")
        
        # Metrics
        lines.append("METRICS:")
        lines.append("-" * 80)
        lines.append(f"Views: {stats.get('views_count', 'no data')}")
        lines.append(f"Likes: {stats.get('likes_count', 'no data')}")
        lines.append(f"Shares: {stats.get('shares_count', 'no data')}")
        lines.append(f"Comments: {stats.get('comments_count', 'no data')}")
        lines.append(f"Saves: {stats.get('savings_count', 'no data')}")
        lines.append("")
        
        # Views breakdown
        views_breakdown = stats.get('views_breakdown', {})
        lines.append("VIEWS BREAKDOWN:")
        lines.append("-" * 80)
        lines.append(f"Views from Followers: {views_breakdown.get('followers_views', 'no data')}")
        lines.append(f"Views from Non-Followers: {views_breakdown.get('non_followers_views', 'no data')}")
        lines.append(f"Followers Percentage: {views_breakdown.get('followers_percentage', 'no data')}")
        lines.append(f"Non-Followers Percentage: {views_breakdown.get('non_followers_percentage', 'no data')}")
        lines.append("")
        
        # Watch time
        avg_watch = stats.get('average_watch_time', {})
        lines.append("WATCH TIME:")
        lines.append("-" * 80)
        lines.append(f"Average Duration: {avg_watch.get('seconds', 'no data')} sec ({avg_watch.get('percentage', 'no data')}%)")
        lines.append(f"Total Watch Time: {stats.get('total_watch_time', 'no data')}")
        lines.append("")
        
        # Ratios
        lines.append("RATIOS:")
        lines.append("-" * 80)
        lines.append(f"Likes to Views: {stats.get('likes_to_views_ratio', 'no data')}")
        lines.append("")
        
        # Traffic sources
        traffic = stats.get('traffic_sources', {})
        lines.append("TRAFFIC SOURCES:")
        lines.append("-" * 80)
        lines.append(f"Profile: {traffic.get('profile', 'no data')}")
        lines.append(f"Feed: {traffic.get('feed', 'no data')}")
        lines.append(f"Reels Tab: {traffic.get('reels_tab', 'no data')}")
        lines.append(f"Stories: {traffic.get('stories', 'no data')}")
        lines.append(f"Explore: {traffic.get('explore', 'no data')}")
        lines.append(f"Other: {traffic.get('other', 'no data')}")
        lines.append("")
        
        # AI Analysis
        ai_analysis = stats.get('ai_analysis', {})
        lines.append("AI ANALYSIS:")
        lines.append("-" * 80)
        lines.append(f"Engagement Assessment: {ai_analysis.get('engagement_assessment', 'no data')}")
        lines.append(f"Completion Rate: {ai_analysis.get('completion_rate', 'no data')}")
        lines.append(f"Virality Potential: {ai_analysis.get('virality_potential', 'no data')}")
        lines.append("")
        
        # Recommendations (now inside ai_analysis)
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 80)
        lines.append(ai_analysis.get('recommendations', 'no data'))
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_results(self, file_name: str, stats: dict):
        """Save results to JSON file"""
        logger.info(f"Saving results for {file_name}")
        try:
            output_file = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".json",
                initialfile=f"stats_{os.path.splitext(file_name)[0]}.json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if output_file:
                logger.debug(f"Saving to file: {output_file}")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
                logger.info(f"Results successfully saved to {output_file}")
                messagebox.showinfo("Success", f"Results saved to:\n{output_file}")
        except Exception as e:
            error_msg = f"Failed to save file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Error", error_msg)
    
    def copy_to_clipboard(self, text: str):
        """Copy text to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Text copied to clipboard")


def main():
    """Launch application"""
    try:
        logger.info("Starting GUI application")
        root = tk.Tk()
        app = InstagramStatsGUI(root)
        root.mainloop()
        logger.info("GUI application closed")
    except Exception as e:
        error_msg = f"Critical error starting application: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        messagebox.showerror("Critical Error", error_msg)
        raise


if __name__ == "__main__":
    main()

