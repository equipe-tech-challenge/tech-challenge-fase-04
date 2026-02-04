# Tech Challenge - Fase 4

Projeto de reconhecimento facial e detecção de expressões em vídeo utilizando **DeepFace** e **face_recognition**.

## Funcionalidades

- Reconhecimento facial a partir de imagens de referência
- Detecção de expressões/emoções em cada frame do vídeo
- Geração de vídeo de saída com anotações (nome + emoção)
- Exportação das emoções detectadas em arquivo `.txt`

## Pré-requisitos

- Python 3.8+
- CMake
- Compilador C++ (necessário para compilar o `dlib`, dependência do `face_recognition`)

### Instalando o CMake e compilador C++

**Windows:**

1. Instale o [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) com o workload "Desenvolvimento para desktop com C++".
2. Instale o CMake via pip:
   ```bash
   pip install cmake
   ```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
```

**macOS:**

```bash
xcode-select --install
brew install cmake
```

## Instalacao

1. Clone o repositorio:
   ```bash
   git clone <url-do-repositorio>
   cd tech-challenge-fase-04
   ```

2. Crie e ative um ambiente virtual (recomendado):
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

3. Instale o CMake (caso ainda nao tenha):
   ```bash
   pip install cmake
   ```

4. Instale as dependencias:
   ```bash
   pip install -r requirements.txt
   ```

> **Nota:** A biblioteca `face_recognition` depende do `dlib`, que precisa ser compilado nativamente. Caso a instalacao falhe, verifique se o CMake e um compilador C++ estao corretamente instalados e disponiveis no PATH do sistema.

## Como usar

1. Coloque as imagens de referencia (rostos conhecidos) na pasta `images/` (formatos `.jpg` ou `.png`).
2. Coloque o video de entrada em `Input/input_video.mp4`.
3. Crie a pasta de saida caso nao exista:
   ```bash
   mkdir Output
   ```
4. Execute o script:
   ```bash
   python Tech_challenge_4.py
   ```

5. O video processado sera salvo em `Output/output_video.mp4` e as emocoes detectadas em `Output/output_video_emocoes.txt`.
