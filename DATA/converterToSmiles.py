import os
import subprocess
import csv
import time

# Especifique o caminho para a pasta
caminho_pasta = "QM_symex"

# Use a função listdir() para obter uma lista de todos os arquivos na pasta
arquivos = os.listdir(caminho_pasta)

# Agora, arquivos conterá uma lista com todos os nomes de arquivos na pasta

print(time.ctime())
with open('QM-symex-DB.csv', 'w', newline='') as csvfile:
    for arquivo in arquivos:
        #comando = f"obabel -ixyz D:\QM_symex\{arquivo} -osmi -OD:\Convertidas\{arquivo.replace(".xyz", "")}.smiles"
        comando = f"obabel -ixyz D:\\QM_symex\\{arquivo} -osmi"

        # Execute o comando e capture a saída
        saida = subprocess.run(comando, shell=True, capture_output=True, text=True)
        smiles = saida.stdout.strip().split()[0]
        smiles_array = smiles.split('|')
        
        with open(f"D:\\QM_symex\\{arquivo}", "r") as arquivo:
            linhas = arquivo.readlines()
            dados_homo = []

            # Flag para indicar se já encontramos a linha "HOMO"
            encontrou_homo = False
            OSs = 0
            OSt = 0

            HomoNumber = ''
                        
            SOrbitals = ''
            TOrbitais = ''

            TOrbital = 0
            TOrbitals = ''

            singlet = ''
            triplet = ''
            # Itera sobre as linhas do arquivo
            for linha in linhas[172736:]:
                # Se encontrar a linha "HOMO", define a flag como True
                if "HOMO" in linha:
                    encontrou_homo = True
                    HomoNumber = linha.split()[1]
                    continue  # Continua para a próxima linha
                # Se a flag estiver definida como True, armazena a linha
                if encontrou_homo:
                    line = linha.strip().split('|')
                    if float(line[1].split()[3]) >= OSs:
                        SOrbital = 0
                        for i in range(len(line)):
                            if len(line[i+2].split()) == 3 and float(line[i+2].split()[2]) >= SOrbital:
                                SOrbital = float(line[i+2].split()[2])
                                SOrbitals = line[i+2]
                            elif len(line[i+2].split()) != 3:
                                break
                            
                        OSs = float(line[1].split()[3])                        
                        singlet = f'{smiles_array[0]} {HomoNumber} {line[0]} {line[1]} {SOrbitals} '

            dados_homo.append(singlet)
            
            writer = csvfile.write(str(dados_homo) + '\n')
print(time.ctime())
