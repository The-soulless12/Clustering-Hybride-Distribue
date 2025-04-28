import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
from clustering import kmeans, kmedoids, hybride_distribue, hybride_distribue_2, accuracy
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

        tk.Label(self.bottom_frame, text="Nombre de classes (K) :").grid(row=0, column=2, padx=10)
        self.entry_k = tk.Entry(self.bottom_frame, width=5)
        self.entry_k.grid(row=0, column=3)

        self.btn_quit = tk.Button(self.bottom_frame, text="Quitter", command=self.quit)
        self.btn_quit.grid(row=0, column=4, padx=10)

        self.label_partitions = tk.Label(self.bottom_frame, text="Nombre de partitions :")
        self.label_partitions.grid(row=1, column=2, padx=5)

        self.entry_partitions = tk.Entry(self.bottom_frame, width=5)
        self.entry_partitions.insert(0, "4")  
        self.entry_partitions.grid(row=1, column=3, padx=5)

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
        self.df = self.df.iloc[:, :-3]
        self.afficher(self.df)

        self.btn_load.grid()
        self.btn_cluster.grid()

    def lancer_clustering(self):
        if self.df is None:
            messagebox.showerror("Erreur", "Dataset manquant. Veuillez charger un fichier avant de lancer le clustering.")
            return

        popup = tk.Toplevel(self)
        popup.title("Choisissez une méthode hybride")
        popup.geometry("300x150")
        popup.configure(bg="#ffe6f0")
        popup.grab_set()

        choix_var = tk.StringVar(value="hybride1")

        tk.Label(popup, text="Sélectionnez la méthode hybride :", bg="#ffe6f0").pack(pady=10)
        tk.Radiobutton(popup, text="Hybride 1", variable=choix_var, value="hybride1", bg="#ffe6f0").pack(anchor="w", padx=20)
        tk.Radiobutton(popup, text="Hybride 2", variable=choix_var, value="hybride2", bg="#ffe6f0").pack(anchor="w", padx=20)

        def valider_choix():
            choix = choix_var.get()
            popup.destroy()

            df_copy1 = self.df.copy()
            df_copy2 = self.df.copy()
            df_copy3 = self.df.copy()

            try:
                K = int(self.entry_k.get())
                if K <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer un nombre entier positif pour K.")
                return

            try:
                n_partitions = int(self.entry_partitions.get())
            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer un nombre valide de partitions.")
                return

            data_np = self.df.iloc[:, :-1].values  
            initial_indices = np.random.choice(len(data_np), K, replace=False)
            initial_centers = data_np[initial_indices]

            # KMeans
            start_kmeans = time.time()
            df_kmeans = kmeans(df_copy1, K=K, initial_centroids=initial_centers)
            time_kmeans = time.time() - start_kmeans

            # KMedoids
            start_kmedoids = time.time()
            data_np2 = df_copy2.select_dtypes(include=[float, int]).values
            initial_medoids_indices = [np.where((data_np2 == center).all(axis=1))[0][0] for center in initial_centers]
            df_kmedoids = kmedoids(df_copy2, K=K, initial_medoids_indices=initial_medoids_indices)
            time_kmedoids = time.time() - start_kmedoids

            # Hybride selon le choix
            start_hybride = time.time()
            if choix == "hybride1":
                df_hybride = hybride_distribue(df_copy3, K=K, n_partitions=n_partitions, initial_centers=initial_centers)
            else:
                df_hybride = hybride_distribue_2(df_copy3, K=K, n_partitions=n_partitions, initial_centers=initial_centers)
            time_hybride = time.time() - start_hybride

            if not any(col in self.tree["columns"] for col in ["K-means", "K-medoid", "Hybride"]):
                self.df["K-means"] = ""
                self.df["K-medoid"] = ""
                self.df["Hybride"] = ""
                self.resultats()

            self.df["K-means"] = df_kmeans["KMeans_Labels"]
            self.df["K-medoid"] = df_kmedoids["KMedoids_Labels"]
            self.df["Hybride"] = df_hybride["Hybrid_Distributed_Labels" if choix == "hybride1" else "Hybrid_Distributed_Labels_2"]

            self.afficher(self.df)

            true_col = None
            for col in self.df.columns:
                if self.df[col].dtype == 'object' and self.df[col].nunique() < 20:
                    true_col = col
                    break

            acc_kmeans = acc_kmedoids = acc_hybride = "Inconnue"

            if true_col:
                try:
                    acc_kmeans = accuracy(self.df[true_col], df_kmeans['KMeans_Labels'])
                except Exception as e:
                    print("Erreur accuracy kmeans :", e)
                try:
                    acc_kmedoids = accuracy(self.df[true_col], df_kmedoids['KMedoids_Labels'])
                except Exception as e:
                    print("Erreur accuracy kmedoids :", e)
                try:
                    acc_hybride = accuracy(self.df[true_col], df_hybride['Hybrid_Distributed_Labels' if choix == "hybride1" else 'Hybrid_Distributed_Labels_2'])
                except Exception as e:
                    print("Erreur accuracy hybride :", e)

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
                
                for widget in self.result_boxes[2].winfo_children():
                    if isinstance(widget, tk.Label) and "Accuracy" in widget.cget("text"):
                        widget.config(text=f"Accuracy: {float(acc_hybride) * 100:.2f}%" if acc_hybride != "Inconnue" else "Accuracy: Inconnue")
                    if isinstance(widget, tk.Label) and "Temps" in widget.cget("text"):
                        widget.config(text=f"Temps: {float(time_hybride):.5f}s")

        tk.Button(popup, text="Valider", command=valider_choix).pack(pady=10)

if __name__ == "__main__":
    app = Interface()
    app.mainloop()