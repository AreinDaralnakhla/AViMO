import subprocess

# Liste des répertoires à traiter
directories = [
    "B5-1", "B5-2", "D1-4", "D10-2", "D10-3", "D11-2", "D11-3", "D12-3", "D4-3", "D4-4", "D5-2", "D7-1", "D7-2", "D7-3", "D8-2"
]

# Les quatre fichiers à convertir (sans l’extension .pkl, qu’on ajoutera dans la boucle)
files = ["nlf-t", "hmr2-t", "multihmr-t", "tokenhmr-t"]

# Paramètres communs à transmettre à convertCapture3D.py
# (au besoin, modifiez-les ou transformez-les en variables si nécessaire)
common_args = ["0", "0", "1", "1", "1", "0", "neutral", ".\\camera_settings.json", "1"]

for directory in directories:
    for f in files:
        input_pkl = f".\\templates-Visualization\\IMPROVE2025\\{directory}\\{f}.pkl"
        output_dir = f".\\templates-Visualization\\IMPROVE2025\\{directory}\\{f}"

        # Commande à exécuter
        cmd = ["python", ".\\convertCapture3D.py", input_pkl, output_dir] + common_args
        
        print("Exécution de la commande :", " ".join(cmd))
        subprocess.run(cmd, shell=True)
