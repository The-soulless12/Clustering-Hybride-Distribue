import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd

class Interface(tk.Tk):
    def __init__(self):
        super().__init__()        

        self.title("Clustering K-Means & K-Medoids")
        self.geometry("800x600")
        self.configure(bg="#ffe6f0") 
        self.resizable(False, False)  
        default_font = ("Comic Sans MS", 11)
        self.option_add("*Font", default_font)

        self.df = None

        navbar = tk.Frame(self, bg="#ffb3cc", height=50)
        navbar.pack(fill=tk.X)

        label_title = tk.Label(navbar, text="K MEANS  •  K MEDOID  •  HYBRIDATION", bg="#ffb3cc", fg="white")
        label_title.pack(pady=10)

        table_frame = tk.Frame(self, bg="#ffe6f0")
        table_frame.place(x=20, y=80, width=760, height=400)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self.tree = ttk.Treeview(table_frame, yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar.config(command=self.tree.yview)

        style = ttk.Style(self)
        style.configure("Treeview", font=("Comic Sans MS", 10), rowheight=25)
        style.configure("Treeview.Heading", font=("Comic Sans MS", 11, "bold"))

        bottom_frame = tk.Frame(self, bg="#ffe6f0")
        bottom_frame.pack(side=tk.BOTTOM, pady=20)

        btn_load = tk.Button(bottom_frame, text="Charger Dataset (Excel)", command=self.load_excel)
        btn_load.grid(row=0, column=0, padx=10)

        btn_cluster = tk.Button(bottom_frame, text="Clustering", command=self.lancer_clustering)
        btn_cluster.grid(row=0, column=1, padx=10)

    def load_excel(self):
        filepath = filedialog.askopenfilename(filetypes=[("Fichiers Excel ou CSV", "*.xlsx *.xls *.csv")])
        if filepath:
            if filepath.endswith(".csv"):
                self.df = pd.read_csv(filepath)
            else:
                self.df = pd.read_excel(filepath)
            
            self.display_dataframe(self.df)

    def display_dataframe(self, df):
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"

        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")

        for _, row in df.iterrows():
            self.tree.insert("", tk.END, values=list(row))

    def lancer_clustering(self):
        if self.df is None:
            messagebox.showerror("Erreur", "Dataset manquant. Veuillez charger un fichier avant de lancer le clustering.")
            return
        
        print("Clustering lancé")

if __name__ == "__main__":
    app = Interface()
    app.mainloop()
