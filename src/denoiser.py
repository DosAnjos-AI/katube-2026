import os
import torch
from df.enhance import enhance, init_df, load_audio, save_audio

class Denoiser:
    def __init__(self, model_name: str = "DeepFilterNet3", device: str = None):
        """
        Inicializa o Denoiser com o modelo DeepFilterNet.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.state, self.df_model, self.model_name = init_df(model_name)
        self.df_sr = 48000  # DeepFilterNet sempre usa 48 kHz
        print(f"[INFO] Modelo {self.model_name} carregado no dispositivo: {self.device}")

    def process_file(self, input_path: str, output_path: str):
        """
        Aplica o denoiser a um único arquivo de áudio.
        """
        audio, _ = load_audio(input_path, sr=self.df_sr)
        enhanced = enhance(self.state, self.df_model, audio)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_audio(output_path, enhanced, self.df_sr)
        print(f"[INFO] Arquivo processado e salvo em: {output_path}")

    def process_batch(self, input_dir: str, output_dir: str):
        """
        Processa todos os arquivos de áudio em um diretório.
        """
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".wav", ".flac", ".ogg")):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".flac")
                self.process_file(input_path, output_path)

        print(f"[INFO] Todos os arquivos foram processados. Resultados em: {output_dir}")

if __name__ == "__main__":
    denoiser = Denoiser(model_name="DeepFilterNet3")

    # Processar um arquivo
    denoiser.process_file(r"C:\Igor\BIA\Alcateia\audios_denoised2\75fK0iwhxdE_segment_0002.flac", "audios_denoised/segmento2_teste.flac")

    # Processar todos os arquivos de um diretório
    #denoiser.process_batch(r"C:\Igor\BIA\Alcateia\Katube_2025_new\audios_download\output\75fK0iwhxdE\segments", "audios_denoised2")
