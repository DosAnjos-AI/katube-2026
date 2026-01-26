#!/usr/bin/env python3
"""
Processador Universal de Legendas para Projeto Katube
Centro de Excelência em Inteligência Artificial (CEIA) - UFG

Módulo para extração e normalização de timestamps de legendas do YouTube.
Suporta múltiplos formatos de entrada com detecção automática:

Formatos Suportados:
    - srv3 (XML): Formato nativo YouTube com timestamps em milissegundos
    - SRT: Formato SubRip com numeração de blocos
    - WebVTT: Web Video Text Tracks (múltiplas variantes)

Processamento Específico:
    - Legendas automáticas (prefixo 'auto_'): Ajuste de timestamps
    - Legendas manuais (prefixo 'manual_'): Preservação de timestamps originais
    
Autor: Marcos dos Anjos
Projeto: Katube - Sistema de Transcrição e Análise de Áudio
"""

import re
from typing import List, Dict, Tuple

# Recebendo o id do audio corrente
id_video= 'QN7gUP7nYhQ'
class WebVTTProcessor:
    """
    Processador universal de legendas com detecção automática de formato.
    
    Atributos:
        filepath (str): Caminho para arquivo de legenda
        content (str): Conteúdo do arquivo carregado
        is_srv3 (bool): Indica se formato é srv3 (XML)
        is_srt (bool): Indica se formato é SRT
        format_type (str): Tipo específico detectado (detailed_with_speakers, simple, etc.)
    
    Métodos Principais:
        extract_segments(): Extrai segmentos com timestamps normalizados
        to_csv(): Exporta segmentos para formato CSV padronizado
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.content = self._read_file()
        self.is_srv3 = self._detect_srv3_format()
        self.is_srt = False if self.is_srv3 else self._detect_srt_format()
        if self.is_srt:
            self.content = self._convert_srt_to_webvtt()
        self.format_type = self._detect_format()
        
    def _detect_srv3_format(self) -> bool:
        """
        Detecta formato srv3 (XML nativo do YouTube).
        
        srv3 é o formato JSON-like em XML usado internamente pelo YouTube
        para armazenar legendas com timestamps em milissegundos e scores
        de confiança por palavra.
        
        Returns:
            bool: True se arquivo é srv3, False caso contrário
        
        Estrutura esperada:
            <?xml version="1.0" encoding="utf-8" ?>
            <timedtext format="3">
        """
        return self.content.strip().startswith('<?xml') and '<timedtext format="3">' in self.content
        
    def _read_file(self) -> str:
        """Lê o arquivo de legenda"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _detect_srt_format(self) -> bool:
        """
        Detecta se é formato SRT clássico
        SRT tem numeração de blocos e vírgula nos timestamps
        """
        # Verificar se tem numeração de blocos e vírgula no timestamp
        srt_pattern = r'^\d+\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}'
        return bool(re.search(srt_pattern, self.content, re.MULTILINE))
    
    def _convert_srt_to_webvtt(self) -> str:
        """
        Converte SRT para WebVTT
        - Remove numeração de blocos
        - Substitui vírgula por ponto nos timestamps
        """
        content = self.content
        
        # Remover numeração de blocos (linhas que são apenas números)
        content = re.sub(r'^\d+\n', '', content, flags=re.MULTILINE)
        
        # Substituir vírgula por ponto nos timestamps
        content = re.sub(
            r'(\d{2}:\d{2}:\d{2}),(\d{3})',
            r'\1.\2',
            content
        )
        
        # Adicionar cabeçalho WEBVTT se não existir
        if not content.startswith('WEBVTT'):
            content = 'WEBVTT\n\n' + content
        
        return content
    
    def _detect_format(self) -> str:
        """
        Detecta automaticamente o formato da legenda
        
        Returns:
            'detailed_with_speakers': Tags <c> + marcadores >>
            'detailed_no_speakers': Tags <c> sem marcadores >>
            'simple': Sem tags <c>, apenas blocos de texto
        """
        has_c_tags = '<c>' in self.content or '</c>' in self.content
        # Detectar tanto &gt;&gt; (HTML) quanto >> literal
        has_speaker_markers = ('&gt;&gt;' in self.content) or ('>>' in self.content and '<c>' in self.content)
        has_nbsp = '&nbsp;' in self.content
        
        if has_c_tags and has_speaker_markers:
            return 'detailed_with_speakers'
        elif has_c_tags and not has_speaker_markers:
            return 'detailed_no_speakers'
        else:
            return 'simple'
    
    @staticmethod
    def timestamp_to_seconds(ts: str) -> float:
        """Converte timestamp para segundos"""
        parts = ts.strip().split(':')
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    
    def _clean_text(self, text: str, format_type: str) -> str:
        """
        Limpa o texto de acordo com o formato
        Remove tags não-verbais como [Música], [Aplausos], etc.
        
        Args:
            text: Texto a ser limpo
            format_type: Tipo de formato detectado
            
        Returns:
            Texto limpo
        """
        # Remover timestamps internos <00:00:00.000>
        text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', text)
        
        # Remover tags <c> e </c>
        text = re.sub(r'</?c>', '', text)
        
        # Substituir &nbsp; por espaço normal
        text = text.replace('&nbsp;', ' ')
        
        # Remover tags não-verbais: [Música], [Aplausos], [Risos], etc.
        # Padrão: qualquer coisa entre colchetes
        text = re.sub(r'\[.*?\]', '', text)
        
        # Normalizar múltiplos espaços
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _is_non_verbal_text(self, text: str) -> bool:
        """
        Verifica se o texto é apenas tag não-verbal ou vazio
        
        Returns:
            True se for apenas tag não-verbal ou vazio
        """
        # Remover espaços
        text = text.strip()
        
        # Vazio
        if not text:
            return True
        
        # Apenas colchetes com conteúdo dentro
        if re.match(r'^\[.*?\]$', text):
            return True
        
        # Após limpar tags, verifica se sobrou algo
        cleaned = re.sub(r'\[.*?\]', '', text).strip()
        if not cleaned:
            return True
        
        return False
    
    def _detect_overlaps(self, segments: List[Dict], filename: str = "") -> List[Dict]:
        """
        Detecta sobreposições de timestamps nos segmentos ORIGINAIS (antes do ajuste)
        
        Args:
            segments: Lista de segmentos com timestamps originais
            filename: Nome do arquivo para relatório
            
        Returns:
            Lista de dicionários com informações sobre sobreposições
        """
        overlaps = []
        
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            fim_atual = self.timestamp_to_seconds(current['fim'])
            inicio_proximo = self.timestamp_to_seconds(next_seg['inicio'])
            
            # Sobreposição ocorre quando fim_atual > inicio_proximo
            if fim_atual > inicio_proximo:
                sobreposicao = fim_atual - inicio_proximo
                overlaps.append({
                    'index': i,
                    'segundo': int(fim_atual),
                    'tempo_fim_atual': current['fim'],
                    'tempo_inicio_proximo': next_seg['inicio'],
                    'sobreposicao_segundos': round(sobreposicao, 3),
                    'trecho_atual': current['texto'][:80] + ('...' if len(current['texto']) > 80 else ''),
                    'trecho_proximo': next_seg['texto'][:80] + ('...' if len(next_seg['texto']) > 80 else '')
                })
        
        return overlaps
    
    def _extract_blocks_detailed(self, max_seconds: float = None) -> List[Dict]:
        """
        Extrai blocos de legendas com formato detalhado (tags <c>)
        Funciona para formatos com ou sem marcadores de locutor
        Filtra tags não-verbais como [Música]
        """
        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})[^\n]*\n((?:.*\n)*?)(?=\n\d{2}:\d{2}:\d{2}\.\d{3}|$)'
        matches = re.finditer(pattern, self.content, re.MULTILINE)
        
        blocos_raw = []
        
        for match in matches:
            tempo_inicio = match.group(1)
            tempo_fim = match.group(2)
            
            # Limitar por tempo se especificado
            if max_seconds and self.timestamp_to_seconds(tempo_inicio) > max_seconds:
                break
            
            texto_bloco = match.group(3).strip()
            linhas = [l.strip() for l in texto_bloco.split('\n') if l.strip()]
            
            for linha in linhas:
                # Pular linhas com apenas timestamps internos ou vazias
                if re.match(r'^<\d{2}:\d{2}:', linha) or linha == ' ':
                    continue
                
                # Verificar se é tag não-verbal ANTES de limpar
                if self._is_non_verbal_text(linha):
                    continue
                
                # Detectar mudança de locutor (>> literal ou &gt;&gt; HTML)
                tem_marcador = ('&gt;&gt;' in linha) or (linha.strip().startswith('>>'))
                
                # Limpar texto
                texto_limpo = self._clean_text(linha, self.format_type)
                
                # Remover marcadores de locutor
                texto_limpo = texto_limpo.replace('&gt;&gt;', '').replace('>>', '').strip()
                
                if texto_limpo:
                    blocos_raw.append({
                        'texto': texto_limpo,
                        'inicio': tempo_inicio,
                        'fim': tempo_fim,
                        'mudanca_locutor': tem_marcador
                    })
        
        return self._consolidate_blocks(blocos_raw)
    
    def _consolidate_blocks(self, blocos_raw: List[Dict]) -> List[Dict]:
        """
        Consolida blocos removendo duplicações consecutivas de texto
        Mantém timestamps e marcadores de locutor
        """
        if not blocos_raw:
            return []
        
        consolidados = []
        ultimo_texto = None
        
        for bloco in blocos_raw:
            texto_atual = bloco['texto']
            
            # Adicionar apenas se for diferente do último
            if texto_atual != ultimo_texto:
                consolidados.append({
                    'texto': texto_atual,
                    'inicio': bloco['inicio'],
                    'fim': bloco['fim'],
                    'comeca_locutor': bloco.get('mudanca_locutor', False) or (len(consolidados) == 0)
                })
                ultimo_texto = texto_atual
        
        return consolidados
    
    def _extract_blocks_srv3(self, max_seconds: float = None) -> List[Dict]:
        """
        Extrai blocos de legendas no formato srv3 (XML do YouTube)
        
        Estrutura srv3:
        - <p t="tempo_ms" d="duracao_ms">texto</p> (legendas manuais)
        - <p t="tempo_ms" d="duracao_ms"><s ac="conf">palavra</s><s t="offset_ms">palavra2</s></p> (automáticas)
        
        Returns:
            Lista de segmentos com timestamps em formato HH:MM:SS.mmm
        """
        import xml.etree.ElementTree as ET
        
        # Parse XML
        try:
            root = ET.fromstring(self.content)
        except Exception as e:
            print(f"Erro ao parsear XML: {e}")
            return []
        
        segments = []
        primeiro_bloco = True
        
        # Procurar tag <body>
        body = root.find('body')
        if body is None:
            return []
        
        # Iterar sobre tags <p> (parágrafos/blocos)
        for p in body.findall('p'):
            # Atributos: t=tempo_inicio_ms, d=duracao_ms
            t_ms = p.get('t')
            d_ms = p.get('d')
            
            if not t_ms or not d_ms:
                continue
            
            t_ms = int(t_ms)
            d_ms = int(d_ms)
            
            # Limitar por tempo
            if max_seconds and (t_ms / 1000.0) > max_seconds:
                break
            
            # Extrair texto
            # Pode ter <s> tags (automática) ou texto direto (manual)
            s_tags = p.findall('s')
            
            if s_tags:
                # Legenda automática: palavra por palavra em <s>
                palavras = []
                for s in s_tags:
                    texto = s.text or ''
                    palavras.append(texto)
                texto_completo = ''.join(palavras).strip()
            else:
                # Legenda manual: texto direto
                texto_completo = ''.join(p.itertext()).strip()
            
            # Verificar se é tag não-verbal
            if self._is_non_verbal_text(texto_completo):
                continue
            
            # Limpar texto
            texto_limpo = self._clean_text(texto_completo, 'srv3')
            
            if not texto_limpo:
                continue
            
            # Converter milissegundos para timestamp
            inicio = self._ms_to_timestamp(t_ms)
            fim = self._ms_to_timestamp(t_ms + d_ms)
            
            # Detectar início de locutor (hífens em manuais)
            comeca_locutor = primeiro_bloco or texto_limpo.startswith('-')
            
            # Remover hífens iniciais
            while texto_limpo.startswith('-'):
                texto_limpo = texto_limpo[1:].strip()
            
            segments.append({
                'texto': texto_limpo,
                'inicio': inicio,
                'fim': fim,
                'comeca_locutor': comeca_locutor
            })
            
            primeiro_bloco = False
        
        return segments
    
    def _ms_to_timestamp(self, ms: int) -> str:
        """Converte milissegundos para timestamp HH:MM:SS.mmm"""
        seconds = ms / 1000.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms_remainder = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms_remainder:03d}"
    
    def _extract_blocks_simple(self, max_seconds: float = None) -> List[Dict]:
        """
        Extrai blocos de legendas com formato simples (sem tags <c>)
        Detecta hífens (-) OU >> no início de linha como marcadores de mudança de locutor
        MANTÉM múltiplas falas do mesmo bloco juntas (não separa hífens do mesmo timestamp)
        Filtra tags não-verbais como [Música]
        """
        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\n((?:.*\n)*?)(?=\n\d{2}:\d{2}:\d{2}\.\d{3}|$)'
        matches = re.finditer(pattern, self.content, re.MULTILINE)
        
        trechos = []
        primeiro_bloco = True
        
        for match in matches:
            tempo_inicio = match.group(1)
            tempo_fim = match.group(2)
            
            # Limitar por tempo se especificado
            if max_seconds and self.timestamp_to_seconds(tempo_inicio) > max_seconds:
                break
            
            texto_bloco = match.group(3).strip()
            
            # Verificar se é apenas tag não-verbal
            if self._is_non_verbal_text(texto_bloco):
                continue
            
            # Detectar se tem marcador >> no início (além de hífen)
            tem_marcador_speaker = texto_bloco.strip().startswith('>>')
            
            # Limpar o texto
            texto_limpo = self._clean_text(texto_bloco, self.format_type)
            texto_limpo = texto_limpo.replace('\n', ' ')
            texto_limpo = re.sub(r'\s+', ' ', texto_limpo)
            
            # Remover >> se tiver
            texto_limpo = texto_limpo.replace('>>', '').strip()
            
            # Remover TODOS os hífens iniciais (podem ser múltiplos)
            # Ex: "- Fala - E aí" → "Fala - E aí" (mantém hífen do meio)
            while texto_limpo.startswith('-'):
                texto_limpo = texto_limpo[1:].strip()
            
            # Verificar novamente após limpeza
            if not texto_limpo or self._is_non_verbal_text(texto_limpo):
                continue
            
            # MANTÉM todo o bloco como um único segmento
            # (não separa por hífens internos)
            trechos.append({
                'texto': texto_limpo,
                'inicio': tempo_inicio,
                'fim': tempo_fim,
                'comeca_locutor': primeiro_bloco or tem_marcador_speaker
            })
            primeiro_bloco = False
        
        return trechos
    
    def _remove_non_speech_blocks_srt(self, segments: List[Dict]) -> List[Dict]:
        """
        Remove blocos sem fala (música, aplausos, pausas, etc) em arquivos SRT
        Também remove 2 linhas antes e 2 depois de cada gap não-verbal
        Ajusta timestamps conectando blocos válidos
        
        Um gap é:
        - Bloco vazio (sem texto)
        - Bloco com apenas tags não-verbais: [Música], [Aplausos], etc
        
        Args:
            segments: Lista de segmentos extraídos
            
        Returns:
            Lista de segmentos filtrados e com timestamps ajustados
        """
        if not segments:
            return segments
        
        # Identificar índices de blocos sem fala (gaps)
        gap_indices = set()
        
        for i, seg in enumerate(segments):
            texto = seg['texto'].strip()
            
            # Bloco sem fala APENAS se:
            # - Vazio OU
            # - Contém apenas tags não-verbais
            if not texto or self._is_non_verbal_text(texto):
                gap_indices.add(i)
        
        # Expandir gaps: adicionar 2 antes e 2 depois de cada gap
        expanded_indices = set()
        
        for gap_idx in gap_indices:
            # Adicionar o próprio gap
            expanded_indices.add(gap_idx)
            
            # 2 antes
            if gap_idx - 1 >= 0:
                expanded_indices.add(gap_idx - 1)
            if gap_idx - 2 >= 0:
                expanded_indices.add(gap_idx - 2)
            
            # 2 depois
            if gap_idx + 1 < len(segments):
                expanded_indices.add(gap_idx + 1)
            if gap_idx + 2 < len(segments):
                expanded_indices.add(gap_idx + 2)
        
        # Filtrar: manter apenas índices NÃO expandidos
        filtered = []
        for i, seg in enumerate(segments):
            if i not in expanded_indices:
                filtered.append(seg)
        
        return filtered
    
    def _adjust_srt_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Ajusta timestamps para legendas automáticas (SRT e srv3).
        
        Problema:
            Legendas automáticas do YouTube incluem pausas e silêncios no
            tempo_fim, resultando em sobreposições de 2-4 segundos entre
            blocos consecutivos.
        
        Solução:
            Para cada segmento N (exceto o último):
                tempo_fim[N] = tempo_inicio[N+1]
            
            Esta abordagem elimina gaps e sobreposições, produzindo
            alinhamento temporal preciso para análise de WER.
        
        Args:
            segments: Lista de segmentos com timestamps originais
            
        Returns:
            Lista de segmentos com timestamps ajustados
            
        Nota:
            Último segmento mantém tempo_fim original pois não há
            segmento posterior para referência.
        """
        if not segments:
            return segments
        
        adjusted = []
        
        for i, seg in enumerate(segments):
            # Para todos exceto o último: usar inicio do próximo como fim
            if i < len(segments) - 1:
                adjusted.append({
                    'texto': seg['texto'],
                    'inicio': seg['inicio'],
                    'fim': segments[i + 1]['inicio'],  # Timestamp do próximo bloco
                    'comeca_locutor': seg.get('comeca_locutor', False)
                })
            else:
                # Último segmento: preservar tempo_fim original
                adjusted.append(seg)
        
        return adjusted
    
    def _is_automatic_subtitle(self) -> bool:
        """
        Verifica se a legenda é automática baseado no nome do arquivo
        Arquivos automáticos: auto_*.txt
        Arquivos manuais: manual_*.txt
        """
        import os
        filename = os.path.basename(self.filepath)
        return filename.startswith('auto_')
       
    
    def extract_segments(self, max_seconds: float = None) -> List[Dict]:
        """
        Extrai segmentos de legenda com timestamps normalizados.
        
        Pipeline de Processamento:
            1. Detecção de formato (srv3 > SRT > WebVTT)
            2. Extração de segmentos específica por formato
            3. Aplicação condicional de ajuste de timestamps:
                - Legendas automáticas (auto_*): Ajuste aplicado
                - Legendas manuais (manual_*): Timestamps preservados
        
        Tratamento por Formato:
            srv3 automático:
                - Parse XML
                - Conversão ms -> HH:MM:SS.mmm
                - Ajuste de timestamps
            
            srv3 manual:
                - Parse XML
                - Conversão ms -> HH:MM:SS.mmm
                - Sem ajuste (timestamps já otimizados)
            
            SRT automático:
                - Conversão para WebVTT
                - Filtro de gaps não-verbais + buffer
                - Ajuste de timestamps
            
            SRT manual:
                - Conversão para WebVTT
                - Extração simples
                - Sem ajuste
            
            WebVTT:
                - Extração palavra-por-palavra (detailed)
                - Ou extração por bloco (simple)
                - Deduplicação automática
        
        Args:
            max_seconds: Limite temporal em segundos. None processa arquivo completo.
            
        Returns:
            Lista de dicionários contendo:
                - texto (str): Transcrição do segmento
                - inicio (str): Timestamp inicial no formato HH:MM:SS.mmm
                - fim (str): Timestamp final no formato HH:MM:SS.mmm
                - comeca_locutor (bool): Indica mudança de locutor
        
        Exemplo:
            >>> processor = WebVTTProcessor('auto_xyz.txt')
            >>> segments = processor.extract_segments()
            >>> segments[0]
            {
                'texto': 'Olá pessoal',
                'inicio': '00:00:01.234',
                'fim': '00:00:03.456',
                'comeca_locutor': True
            }
        """
        # srv3 tem prioridade máxima de detecção
        if self.is_srv3:
            segments = self._extract_blocks_srv3(max_seconds)
            
            # Ajuste condicional baseado em convenção de nomenclatura
            if self._is_automatic_subtitle():
                segments = self._adjust_srt_timestamps(segments)
            
            return segments
        
        # WebVTT: detecção de subtipo (detailed vs simple)
        if self.format_type in ['detailed_with_speakers', 'detailed_no_speakers']:
            segments = self._extract_blocks_detailed(max_seconds)
        else:
            segments = self._extract_blocks_simple(max_seconds)
        
        # Processamento específico para SRT
        if self.is_srt:
            # 1. Remoção de gaps não-verbais com buffer de contexto
            segments = self._remove_non_speech_blocks_srt(segments)
            
            # 2. Ajuste de timestamps (apenas automáticos)
            if self._is_automatic_subtitle():
                segments = self._adjust_srt_timestamps(segments)
        
        return segments
    
    def to_csv(self, output_path: str, max_seconds: float = None, separator: str = '|'):
        """
        Salva os segmentos em formato CSV
        
        Args:
            output_path: Caminho do arquivo de saída
            max_seconds: Limite de tempo em segundos (None = arquivo completo)
            separator: Separador de colunas (padrão: |)
        """
        segments = self.extract_segments(max_seconds)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f'Trecho{separator}tempo_inicio{separator}tempo_fim{separator}comeca_locutor\n')
            for seg in segments:
                # Escapar separador no texto
                texto = seg['texto'].replace(separator, f'\\{separator}')
                comeca = seg.get('comeca_locutor', False)
                f.write(f"{texto}{separator}{seg['inicio']}{separator}{seg['fim']}{separator}{comeca}\n")
        
        return len(segments)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Caminhos relativos ao diretorio do script
    script_dir = Path(__file__).parent
    id_video_input= "../arquivos/temp/" + id_video + "/01-arquivos_originais"
    id_video_output= "../arquivos/temp/" + id_video + "/01-arquivos_originais"
    input_dir = script_dir / id_video_input
    output_dir = script_dir / id_video_output
    
    # Validacao: diretorio de entrada existe
    if not input_dir.exists():
        print(f"ERRO: Pasta de input nao encontrada: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"ERRO: Input nao e uma pasta: {input_dir}")
        sys.exit(1)
    
    # Busca por arquivo .txt no diretorio de entrada
    txt_files = list(input_dir.glob('*.txt'))
    
    if not txt_files:
        print(f"ERRO: Nenhum arquivo .txt encontrado em: {input_dir}")
        sys.exit(1)
    
    if len(txt_files) > 1:
        print(f"AVISO: {len(txt_files)} arquivos .txt encontrados. Processando o primeiro.")
    
    # Seleciona primeiro arquivo encontrado
    input_file = txt_files[0]
    
    # Cria diretorio de output se nao existir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome do output: mesmo nome do input, extensao .csv
    output_file = output_dir / input_file.with_suffix('.csv').name
    
    # Log de processamento
    print(f"Input:  {input_file.relative_to(script_dir)}")
    print(f"Output: {output_file.relative_to(script_dir)}")
    print(f"\nProcessando...")
    
    try:
        processor = WebVTTProcessor(str(input_file))
        processor.to_csv(str(output_file))
        
        segments = processor.extract_segments()
        
        # Identificacao de tipo: automatico ou manual
        tipo_arquivo = "AUTOMATICO" if processor._is_automatic_subtitle() else "MANUAL"
        formato = "srv3" if processor.is_srv3 else ("SRT" if processor.is_srt else "WebVTT")
        
        print(f"\nSucesso!")
        print(f"   Tipo: {tipo_arquivo}")
        print(f"   Formato: {formato}")
        print(f"   Segmentos: {len(segments)}")
        print(f"   CSV salvo em: {output_file.relative_to(script_dir)}")
        
    except Exception as e:
        print(f"\nERRO ao processar: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)