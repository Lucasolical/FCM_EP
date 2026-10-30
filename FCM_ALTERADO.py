import time
import random
from PIL import Image
import matplotlib.pyplot as plt

def fcm_processar_imagem(img, K, m=2.0, max_iter=100, epsilon=0.01):
    """
    Executa o pipeline completo do FCM Puro com proteções contra índice inválido e divisão por zero.
    """
    largura, altura = img.size

    # Extração segura dos pixels da imagem
    pixels_2d = list(img.getdata())

    X = []
    for p in pixels_2d:
        X.append([float(p[0]), float(p[1]), float(p[2])])
        
    N = len(X)
    canais = 3

    paleta_cores = [
        [255, 0, 0],     # Cluster 0: Vermelho
        [0, 255, 0],     # Cluster 1: Verde
        [0, 0, 255],     # Cluster 2: Azul
        [255, 255, 0],   # Cluster 3: Amarelo
        [255, 0, 255],   # Cluster 4: Magenta
        [0, 255, 255]    # Cluster 5: Ciano
    ]

    U = []
    for i in range(N):
        linha = []
        soma_linha = 0.0
        for j in range(K):
            val = random.random()
            linha.append(val)
            soma_linha += val
        
        for j in range(K):
            if soma_linha == 0.0:
                linha[j] = 1.0 / K
            else:
                linha[j] = linha[j] / soma_linha
        U.append(linha)
        
    # 4. LAÇO PRINCIPAL DO FCM (Puro)
    for iteracao in range(max_iter):
        
        C = []
        for j in range(K):
            numerador = [0.0] * canais
            denominador = 0.0
            
            for i in range(N):
                u_m = U[i][j] ** m
                denominador += u_m
                
                for c in range(canais):
                    numerador[c] += u_m * X[i][c]
            
            centroide_j = []
            for c in range(canais):
                if denominador == 0.0:
                    centroide_j.append(0.0)
                else:
                    centroide_j.append(numerador[c] / denominador)
            C.append(centroide_j)
            
        U_nova = []
        max_diff = 0.0
        
        # ====================================================================
        # APLICAÇÃO DA MISSÃO 6: LOOP FISSION (DIVÓRCIO DE LAÇOS)
        # 
        # Modificação 1: Criamos uma estrutura intermediária na RAM para 
        # armazenar todas as distâncias calculadas antes de prosseguir.
        # ====================================================================
        matriz_distancias = []
        
        # Primeiro Laço Isolado: Varre a RAM apenas para computar e salvar distâncias
        for i in range(N):
            distancias = [0.0] * K
            for j in range(K):
                soma_quadrados = 0.0
                for c in range(canais):
                    diff = X[i][c] - C[j][c]
                    soma_quadrados += diff * diff
                distancias[j] = soma_quadrados ** 0.5 
            matriz_distancias.append(distancias)

        # Segundo Laço Isolado: Executa exclusivamente o cálculo da lógica Fuzzy
        # utilizando o array temporário gerado no passo anterior.
        for i in range(N):
            linha_nova = [0.0] * K
            distancias = matriz_distancias[i] # Recupera as distâncias pré-calculadas
                
            for j in range(K):
                if distancias[j] == 0.0:
                    linha_nova[j] = 1.0
                    for k in range(K):
                        if k != j: 
                            linha_nova[k] = 0.0
                    break
                    
                soma_dist = 0.0
                for k in range(K):
                    d_k = distancias[k] if distancias[k] > 0.0 else 1e-10
                    razao = distancias[j] / d_k
                    soma_dist += razao ** (2.0 / (m - 1.0))
                
                pertinencia = 1.0 / soma_dist if soma_dist > 0.0 else (1.0 / K)
                linha_nova[j] = pertinencia
                
                diff_u = pertinencia - U[i][j]
                if diff_u < 0: 
                    diff_u = -diff_u
                if diff_u > max_diff: 
                    max_diff = diff_u
                
            U_nova.append(linha_nova)
            
        # O U TEM QUE SER ATUALIZADO AQUI FORA DO LAÇO DE PIXELS!
        U = U_nova
            
        if max_diff < epsilon:
            print(f"-> Convergiu na iteração {iteracao}!")
            break
            
    # 5. FASE 5: COLORIZAÇÃO POR PERTINÊNCIA MÁXIMA
    pixels_segmentados_lista = []
    for i in range(N):
        maior_pertinencia = -1.0
        cluster_vencedor = 0
        for j in range(K):
            if U[i][j] > maior_pertinencia:
                maior_pertinencia = U[i][j]
                cluster_vencedor = j
                
        cor = paleta_cores[cluster_vencedor % len(paleta_cores)]
        pixels_segmentados_lista.append((cor[0], cor[1], cor[2]))
        
    # 6. RESHAPE MANUAL E SALVAMENTO
    imagem_final = Image.new("RGB", (largura, altura))
    imagem_final.putdata(pixels_segmentados_lista)

    return C, U, imagem_final

### ==============================================================
### MAIN
### ==============================================================
if __name__ == "__main__":
    imagem_entrada = "./minhas_imagens/bros.png"
    K_clusters = 4                        
    fator_m = 2.0                         
    iteracoes_max = 100                    
    limite_epsilon = 0.001                

    print(f"--- INICIANDO PROCESSAMENTO DE BANCADA ---")
    print(f"Figura de Entrada: {imagem_entrada}")
    print(f"Clusters (K): {K_clusters}")

    try:
        imagem = Image.open(imagem_entrada).convert("RGB")

        plt.figure(figsize=(8, 8))
        plt.title(f"Imagem Original")
        plt.imshow(imagem)
        plt.axis('off')
        plt.show()
        
        t0 = time.perf_counter()
        
        centroides, pertinencias, imagem_segmentada = fcm_processar_imagem(
            img=imagem,
            K=K_clusters,
            m=fator_m,
            max_iter=iteracoes_max,
            epsilon=limite_epsilon
        )
        
        t1 = time.perf_counter()
        tempo_total = t1 - t0
        
        print(f"\n[SUCESSO] Processamento finalizado com folga!")
        print(f"Tempo Total de Execução (Bancada): {tempo_total:.4f} segundos")
        
        plt.figure(figsize=(8, 8))
        plt.title(f"FCM Puro Segmentado - {K_clusters} Clusters")
        plt.imshow(imagem_segmentada)
        plt.axis('off')
        plt.show()
        
    except FileNotFoundError:
        print(f"\n[ERRO] O arquivo '{imagem_entrada}' não foi encontrado.")
    except Exception as e:
        print(f"\n[ERRO INESPERADO] Ocorreu uma falha durante o processamento: {e}")
