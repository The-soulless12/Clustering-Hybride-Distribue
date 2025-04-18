import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
from clustering import kmeans, kmedoids, accuracy
import time

class Interface(tk.Tk):
    def __init__(self):
        super().__init__()        

        self.title("Clustering Application")
        self.geometry("800x600")
        self.configure(bg="#ffe6f0") 
        self.resizable(False, False)  
        default_font = ("Comic Sans MS", 11)
        self.option_add("*Font", default_font)

        self.df = None
        self.result_boxes = []

        navbar = tk.Frame(self, bg="#ffb3cc", height=50)
        navbar.pack(fill=tk.X)

        label_title = tk.Label(navbar, text="K MEANS  •  K MEDOID  •  HYBRIDATION", bg="#ffb3cc", fg="white")
        label_title.pack(pady=10)

        self.table_frame = tk.Frame(self, bg="#ffe6f0")
        self.table_frame.place(x=20, y=80, width=760, height=400)

        scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self.tree = ttk.Treeview(self.table_frame, yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar.config(command=self.tree.yview)

        style = ttk.Style(self)
        style.configure("Treeview", font=("Comic Sans MS", 10), rowheight=25)
        style.configure("Treeview.Heading", font=("Comic Sans MS", 11, "bold"))

        self.bottom_frame = tk.Frame(self, bg="#ffe6f0")
        self.bottom_frame.pack(side=tk.BOTTOM, pady=20)

        self.btn_load = tk.Button(self.bottom_frame, text="Charger Dataset (Excel)", command=self.charger_excel)
        self.btn_load.grid(row=0, column=0, padx=10)

        self.btn_cluster = tk.Button(self.bottom_frame, text="Clustering", command=self.lancer_clustering)
        self.btn_cluster.grid(row=0, column=1, padx=10)

    def charger_excel(self):
        filepath = filedialog.askopenfilename(filetypes=[("Fichiers Excel ou CSV", "*.xlsx *.xls *.csv")])
        if filepath:
            if filepath.endswith(".csv"):
                self.df = pd.read_csv(filepath)
            else:
                self.df = pd.read_excel(filepath)
            
            self.afficher(self.df)

    def afficher(self, df):
        for i in self.tree.get_children():
            self.tree.delete(i)

        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"

        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")

        for _, row in df.iterrows():
            self.tree.insert("", tk.END, values=list(row))

    def resultats(self):
        nouvelles_colonnes = ["K-means", "K-medoid", "Hybride"]
        anciennes_colonnes = list(self.tree["columns"])
        toutes_colonnes = anciennes_colonnes + nouvelles_colonnes
        self.tree.configure(columns=toutes_colonnes)
        self.tree["show"] = "headings" 

        for col in toutes_colonnes:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")

        for i, item in enumerate(self.tree.get_children()):
            valeurs_ligne = list(self.tree.item(item)["values"])
            valeurs_ligne += ["", "", ""]
            self.tree.item(item, values=valeurs_ligne)

        self.btn_load.grid_remove()
        self.btn_cluster.grid_remove()

        self.Resultats = tk.Frame(self, bg="#ffe6f0")
        self.Resultats.place(x=20, y=490, width=760)

        labels = ["K-Means", "K-Medoids", "Hybride"]
        for i, label in enumerate(labels):
            box = tk.Frame(self.Resultats, bg="#ffccd5", width=180, height=100)
            box.grid(row=0, column=i, padx=20)
            box.pack_propagate(False)

            tk.Label(box, text=label, bg="#ffccd5", fg="#4d0039", font=("Comic Sans MS", 12, "bold")).pack(pady=10)
            tk.Label(box, text="Accuracy: ", bg="#ffccd5").pack()
            tk.Label(box, text="Temps: ", bg="#ffccd5").pack()

            self.result_boxes.append(box)

        self.btn_retour = tk.Button(self.Resultats, text="Retour", command=self.revenir_accueil)
        self.btn_retour.place(x=680, y=30)
    
    def revenir_accueil(self):
        if hasattr(self, "Resultats"):
            self.Resultats.destroy()

        self.result_boxes = []

        self.btn_load.grid()
        self.btn_cluster.grid()

        if self.df is not None:
            self.afficher(self.df)

    def lancer_clustering(self):
        if self.df is None:
            messagebox.showerror("Erreur", "Dataset manquant. Veuillez charger un fichier avant de lancer le clustering.")
            return

        df_copy1 = self.df.copy()
        df_copy2 = self.df.copy()

        start_kmeans = time.time()
        df_kmeans = kmeans(df_copy1, K=3)
        time_kmeans = time.time() - start_kmeans

        start_kmedoids = time.time()
        df_kmedoids = kmedoids(df_copy2, K=3)
        time_kmedoids = time.time() - start_kmedoids

        if not any(col in self.tree["columns"] for col in ["K-means", "K-medoid", "Hybride"]):
            self.df["K-means"] = ""
            self.df["K-medoid"] = ""
            self.df["Hybride"] = ""
            self.resultats()

        self.df["K-means"] = df_kmeans["KMeans_Labels"]
        self.df["K-medoid"] = df_kmedoids["KMedoids_Labels"]
        self.afficher(self.df)

        true_col = None
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and self.df[col].nunique() < 20:
                true_col = col
                break

        acc_kmeans = "Inconnue"
        acc_kmedoids = "Inconnue"

        if true_col:
            try:
                acc_kmeans = accuracy(self.df[true_col], df_kmeans['KMeans_Labels'])
            except Exception as e:
                print("Erreur accuracy kmeans :", e)
            try:
                acc_kmedoids = accuracy(self.df[true_col], df_kmedoids['KMedoids_Labels'])
            except Exception as e:
                print("Erreur accuracy kmedoids :", e)

        if self.result_boxes:
            for widget in self.result_boxes[0].winfo_children():
                if isinstance(widget, tk.Label) and "Accuracy" in widget.cget("text"):
                    widget.config(text=f"Accuracy: {float(acc_kmeans) * 100:.2f}%")
                if isinstance(widget, tk.Label) and "Temps" in widget.cget("text"):
                    widget.config(text=f"Temps: {float(time_kmeans):.5f}s")
            
            for widget in self.result_boxes[1].winfo_children():
                if isinstance(widget, tk.Label) and "Accuracy" in widget.cget("text"):
                    widget.config(text=f"Accuracy: {float(acc_kmedoids) * 100:.2f}%")
                if isinstance(widget, tk.Label) and "Temps" in widget.cget("text"):
                    widget.config(text=f"Temps: {float(time_kmedoids):.5f}s")

if __name__ == "__main__":
    app = Interface()
    app.mainloop()