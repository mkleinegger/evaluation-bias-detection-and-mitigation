# Work in progress
also:

am besten du erastellst ein conda env im home verzeichnis (damit er nicht beim nächsten mal gelöscht ist).. nur derm /home folder wird persisitiert:

conda create --prefix ./.ba-thesis  python=3.9

dann:

conda env activaten, dann peotry install und dann zum schluss das enviornment als kernel hinzufügen:

python -m ipykernel install --user --name=ba-thesis (während du im env bist)