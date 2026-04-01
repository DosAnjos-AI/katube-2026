#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# (C) 2021 Frederico Oliveira fred.santos.oliveira(at)gmail.com
#
#
import argparse
from os import makedirs
from os.path import join, exists, dirname
from textdistance import levenshtein
from tqdm import tqdm


def remove_punctuations(sentence):
    """
    Removes punctuations and unwanted characters from a sentence.
    """
    punctuations = '''—!()-[]{};:'"\\,<>./?@#$%^&*_~'''
    sentence_with_no_punct = ""
    for char in sentence:
       if char not in punctuations:
           sentence_with_no_punct = sentence_with_no_punct + char
    return sentence_with_no_punct.strip()


def clear_sentences(sentence):
    """
    Converts the sentence to lowercase and removes unwanted characters.
    """
    sentence = sentence.lower()
    clean_sentence = remove_punctuations(sentence)
    return clean_sentence


def create_validation_file(input_file1, input_file2, prefix_filepath, output_file):
    """
    Given two files containing different transcriptions of audio files, this function calculates the similarity (levenshtein distance) between the sentences,
    saving the result in a third file.

        Parameters:
        input_file1 (str): First filepath. The contents of the file must follow the template: "filename | text"
        input_file2 (str): Second filepath. The contents of the file must follow the template: "filename | text"
        prefix_filepath: Prefix to be added to the file path within the output file.

        Returns:
        output_file (str): Returns output filepath. The content of the file follows the template: prefix_filepath/filename | text1 | text2 | similarity
    """

    # Loads the contents of the first input file
    try:
        with open(input_file1, encoding='utf-8') as f:
            content_file1 = f.readlines()

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected!")
        exit()

    except IOError:
      print("Error: File {} does not appear to exist.".format(input_file1))
      return False

    # Loads the contents of the second input file
    try:
        with open(input_file2, encoding='utf-8') as g:
            content_file2 = g.readlines()

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected!")
        exit()

    except IOError:
      print("Error: File {} does not appear to exist.".format(input_file2))
      return False

    # Both files must be the same length, otherwise there is an error.
    if not (len(content_file1) == len(content_file2)):
        print("Error: length File {} not igual to File {}.".format(content_file1, content_file2))
        return False

    # Checks if the output folder exists
    output_folderpath = dirname(output_file)

    if not(exists(output_folderpath)):
        makedirs(output_folderpath)

    # Saves the result to the output file.
    try:
        o_file = open(output_file, 'w', encoding='utf-8')

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected!")
        exit()

    except IOError:
        print("Error: creating File {} problem.".format(output_file))
        return False

    # Iterate over the two files content simultaneously to calculate the similarity between the sentences.
    else:
        separator = '|'
        header = separator.join(['filename', 'subtitle', 'transcript', 'similarity'])
        o_file.write(header + '\n')

        # Input files must be csv files with the character "|" as a separator: filename | text
        for line1, line2 in tqdm(zip(content_file1, content_file2), total=len(content_file1)):

            file1, text1 = line1.split('|')
            file2, text2 = line2.split('|')

            # Clears sentences by removing unwanted characters.
            clean_text1 = clear_sentences(text1)
            clean_text2 = clear_sentences(text2)
            filepath = join(prefix_filepath, file1)

            # Calculates the levenshtein distance to define the normalized similarity (0-1) between two sentences.
            l = levenshtein.normalized_similarity(clean_text1, clean_text2)

            # Defines the output content and writes to a file.
            line = separator.join([filepath, text1.strip(), text2.strip(), str(l)])           
            o_file.write(line + '\n')

    finally:
        o_file.close()

    return True


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--input_file1', default='metadata1.csv', help='Input first filename')
    parser.add_argument('--input_file2', default='metadata2.csv', help='Input second filename')
    parser.add_argument('--prefix', default='', help='Prefix to filename on metadata output file.')
    parser.add_argument('--output_dir', default='output', help='Directory to save distances')
    parser.add_argument('--output_file', default='validation.csv', help='Output file with the template: "filename, text1, text2, similarity"')

    args = parser.parse_args()

    input_path_file1 = join(args.base_dir, args.input_file1)
    input_path_file2 = join(args.base_dir, args.input_file2)
    output_path_file = join(args.base_dir, args.output_dir, args.output_file)

    create_validation_file(input_path_file1, input_path_file2, args.prefix, output_path_file)


if __name__ == "__main__":
    main()#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validador de Transcrições usando Distância Levenshtein
Compara wav2vec2 vs whisper e salva aprovados em CSV cumulativo
Versão KISS - simples e funcional
"""

import json
import os
import csv
import glob
from datetime import datetime
from textdistance import levenshtein


class ValidadorTranscricao:
    """
    Validador de transcrições baseado em similaridade Levenshtein
    """
    
    def __init__(self, pasta_jsons, threshold=0.85, csv_saida="validacao_transcricoes.csv"):
        """
        Inicializa o validador
        
        Args:
            pasta_jsons (str): Caminho da pasta contendo os JSONs
            threshold (float): Limite mínimo de similaridade (0.0 a 1.0)
            csv_saida (str): Nome do arquivo CSV de saída
        """
        self.pasta_jsons = pasta_jsons
        self.threshold = threshold
        self.csv_saida = os.path.join(pasta_jsons, csv_saida)
        self.colunas_csv = [
            'filename', 
            'wav2vec2_original', 
            'wav2vec2_normalized', 
            'whisper_original', 
            'whisper_normalized', 
            'similarity'
        ]
        
    def calcular_similaridade(self, texto1, texto2):
        """
        Calcula similaridade Levenshtein normalizada entre dois textos
        
        Args:
            texto1 (str): Primeiro texto
            texto2 (str): Segundo texto
            
        Returns:
            float: Similaridade normalizada (0.0 a 1.0)
        """
        if not texto1 or not texto2:
            return 0.0
            
        # Remove espaços extras para comparação mais justa
        clean_texto1 = texto1.strip()
        clean_texto2 = texto2.strip()
        
        if not clean_texto1 or not clean_texto2:
            return 0.0
            
        # Calcula similaridade normalizada
        similarity = levenshtein.normalized_similarity(clean_texto1, clean_texto2)
        return similarity
    
    def buscar_arquivos_json(self):
        """
        Busca todos os arquivos *_normalized_text.json na pasta
        
        Returns:
            list: Lista de caminhos dos arquivos JSON encontrados
        """
        padrao = os.path.join(self.pasta_jsons, "*_normalized_text.json")
        arquivos = glob.glob(padrao)
        return arquivos
    
    def processar_json(self, caminho_json):
        """
        Processa um arquivo JSON e retorna dados dos segmentos aprovados
        
        Args:
            caminho_json (str): Caminho do arquivo JSON
            
        Returns:
            list: Lista de dicionários com dados dos segmentos aprovados
        """
        try:
            with open(caminho_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar {caminho_json}: {e}")
            return []
        
        normalized_pairs = data.get('normalized_pairs', {})
        if not normalized_pairs:
            print(f"Nenhum par normalizado encontrado em {caminho_json}")
            return []
        
        aprovados = []
        total_pares = len(normalized_pairs)
        aprovados_count = 0
        
        print(f"Processando {total_pares} pares de {os.path.basename(caminho_json)}")
        
        for segment_id, pair_data in normalized_pairs.items():
            # Extrai textos normalizados para comparação
            wav2vec2_norm = pair_data.get('wav2vec2_normalized', '').strip()
            whisper_norm = pair_data.get('whisper_normalized', '').strip()
            
            # Extrai textos originais para salvar
            wav2vec2_orig = pair_data.get('wav2vec2_original', '').strip()
            whisper_orig = pair_data.get('whisper_original', '').strip()
            
            # Extrai filename
            filename = pair_data.get('segment_filename', f"{segment_id}.wav")
            
            # Verifica se textos são válidos
            if not wav2vec2_norm or not whisper_norm:
                continue
                
            # Calcula similaridade
            similarity = self.calcular_similaridade(wav2vec2_norm, whisper_norm)
            
            # Verifica se passa no threshold
            if similarity >= self.threshold:
                aprovados_count += 1
                
                aprovado = {
                    'filename': filename,
                    'wav2vec2_original': wav2vec2_orig,
                    'wav2vec2_normalized': wav2vec2_norm,
                    'whisper_original': whisper_orig,
                    'whisper_normalized': whisper_norm,
                    'similarity': round(similarity, 6)
                }
                aprovados.append(aprovado)
        
        taxa_aprovacao = (aprovados_count / total_pares * 100) if total_pares > 0 else 0
        print(f"  Aprovados: {aprovados_count}/{total_pares} ({taxa_aprovacao:.1f}%)")
        
        return aprovados
    
    def salvar_csv_cumulativo(self, novos_dados):
        """
        Salva dados no CSV de forma cumulativa (append)
        
        Args:
            novos_dados (list): Lista de dicionários com novos dados
        """
        if not novos_dados:
            print("Nenhum dado novo para salvar")
            return
            
        # Verifica se arquivo existe para decidir se escreve header
        arquivo_existe = os.path.exists(self.csv_saida)
        
        try:
            with open(self.csv_saida, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.colunas_csv)
                
                # Escreve header apenas se arquivo não existe
                if not arquivo_existe:
                    writer.writeheader()
                    print(f"Criado novo CSV: {self.csv_saida}")
                
                # Escreve todos os novos dados
                for row in novos_dados:
                    writer.writerow(row)
                
                print(f"Adicionados {len(novos_dados)} registros ao CSV")
                
        except Exception as e:
            print(f"Erro ao salvar CSV: {e}")
    
    def validar_todos_jsons(self):
        """
        Processa todos os JSONs da pasta e salva resultados
        
        Returns:
            dict: Estatísticas da validação
        """
        print("=" * 60)
        print("VALIDADOR DE TRANSCRIÇÕES - SIMILARIDADE LEVENSHTEIN")
        print("=" * 60)
        print(f"Pasta: {self.pasta_jsons}")
        print(f"Threshold: {self.threshold}")
        print(f"CSV saída: {self.csv_saida}")
        print("=" * 60)
        
        # Busca arquivos JSON
        arquivos_json = self.buscar_arquivos_json()
        
        if not arquivos_json:
            print("Nenhum arquivo *_normalized_text.json encontrado!")
            return {'arquivos_processados': 0, 'total_aprovados': 0}
        
        print(f"Encontrados {len(arquivos_json)} arquivos JSON")
        print("-" * 60)
        
        # Processa todos os arquivos
        todos_aprovados = []
        arquivos_processados = 0
        
        for arquivo_json in arquivos_json:
            aprovados = self.processar_json(arquivo_json)
            if aprovados:
                todos_aprovados.extend(aprovados)
                arquivos_processados += 1
            print("-" * 40)
        
        # Salva resultados no CSV
        self.salvar_csv_cumulativo(todos_aprovados)
        
        # Estatísticas finais
        stats = {
            'arquivos_encontrados': len(arquivos_json),
            'arquivos_processados': arquivos_processados,
            'total_aprovados': len(todos_aprovados),
            'threshold_usado': self.threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        print("=" * 60)
        print("RESUMO DA VALIDAÇÃO:")
        print(f"  Arquivos JSON encontrados: {stats['arquivos_encontrados']}")
        print(f"  Arquivos processados: {stats['arquivos_processados']}")
        print(f"  Total de aprovados: {stats['total_aprovados']}")
        print(f"  Threshold usado: {stats['threshold_usado']}")
        print(f"  CSV atualizado: {self.csv_saida}")
        print("=" * 60)
        
        return stats


def main():
    """
    Função principal para execução via linha de comando
    """
    print("VALIDADOR DE TRANSCRIÇÕES POR SIMILARIDADE")
    print("Compara wav2vec2 vs whisper usando Levenshtein")
    print("=" * 50)
    
    # Input do usuário
    pasta = input("Digite o caminho da pasta com os JSONs: ").strip()
    
    if not os.path.exists(pasta):
        print(f"Erro: Pasta não encontrada: {pasta}")
        return
    
    try:
        threshold_input = input("Digite o threshold (0.0 a 1.0, padrão 0.85): ").strip()
        threshold = float(threshold_input) if threshold_input else 0.85
        
        if not 0.0 <= threshold <= 1.0:
            print("Threshold deve estar entre 0.0 e 1.0. Usando padrão 0.85")
            threshold = 0.85
            
    except ValueError:
        print("Valor inválido. Usando threshold padrão 0.85")
        threshold = 0.85
    
    # Cria validador e executa
    validador = ValidadorTranscricao(pasta, threshold)
    stats = validador.validar_todos_jsons()
    
    if stats['total_aprovados'] > 0:
        print(f"\n✅ Validação concluída com sucesso!")
        print(f"Verifique o arquivo: {validador.csv_saida}")
    else:
        print(f"\n⚠️  Nenhum segmento aprovado com threshold {threshold}")


if __name__ == "__main__":
    main()