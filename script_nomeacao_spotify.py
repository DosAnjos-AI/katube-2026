"""
Reorganizador de audios.

Para cada audio encontrado recursivamente em PASTA_INPUT, cria uma pasta
com o hash MD5 do caminho completo como ID dentro de PASTA_OUTPUT e move
o arquivo renomeado com o mesmo hash. Gera/atualiza um CSV de relacao
entre hash, nome original e caminhos.

Configuracao do usuario:
    - PASTA_INPUT   : pasta raiz onde os audios serao buscados recursivamente
    - PASTA_OUTPUT  : pasta de destino
    - PASTA_RELACAO : pasta onde o arquivo relacao_ids.csv sera salvo
    - EXTENSAO      : extensao alvo (ex: ".flac", ".wav", ".ogg")
"""

import csv
import hashlib
import logging
import shutil
from pathlib import Path

# =============================================================================
# CONFIGURACAO DO USUARIO
# =============================================================================

PASTA_INPUT   = Path("/home/ubuntu/experimentos_fred/downloaded_segments_3")
PASTA_OUTPUT  = Path("/home/ubuntu/katube_spotify_2026/arquivos/audios")
PASTA_RELACAO = Path("/home/ubuntu/katube_spotify_2026/Dataset_Processado_Spotify2")
EXTENSAO      = ".ogg"

# =============================================================================
# CONSTANTES
# =============================================================================

NOME_RELACAO         = "relacao_ids.csv"
SEPARADOR            = "|"
COLUNAS              = ["hash", "nome_audio", "caminho_origem", "caminho_destino"]
TAMANHO_MINIMO_BYTES = 10

# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def gerar_hash(caminho: Path) -> str:
    """Gera hash MD5 do caminho completo do arquivo."""
    return hashlib.md5(str(caminho).encode()).hexdigest()


def carregar_hashes_existentes(arquivo_relacao: Path) -> set[str]:
    """Carrega hashes ja registrados no CSV para evitar duplicatas."""
    if not arquivo_relacao.exists():
        return set()

    hashes = set()
    with arquivo_relacao.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=SEPARADOR)
        for linha in reader:
            hashes.add(linha["hash"])
    return hashes


def registrar_relacao(arquivo_relacao: Path, registros: list[dict]) -> None:
    """Cria ou faz append no CSV de relacao de IDs."""
    arquivo_existe = arquivo_relacao.exists()

    with arquivo_relacao.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS, delimiter=SEPARADOR)

        if not arquivo_existe:
            writer.writeheader()
            log.info("Arquivo de relacao criado: %s", arquivo_relacao)

        writer.writerows(registros)


def main() -> None:
    if not PASTA_INPUT.exists():
        log.error("Pasta de input nao encontrada: %s", PASTA_INPUT)
        return

    PASTA_OUTPUT.mkdir(parents=True, exist_ok=True)
    PASTA_RELACAO.mkdir(parents=True, exist_ok=True)

    arquivo_relacao   = PASTA_RELACAO / NOME_RELACAO
    hashes_existentes = carregar_hashes_existentes(arquivo_relacao)

    if hashes_existentes:
        log.info("Hashes ja registrados no CSV: %d", len(hashes_existentes))

    arquivos = sorted(PASTA_INPUT.rglob(f"*{EXTENSAO}"))

    if not arquivos:
        log.info("Nenhum arquivo '%s' encontrado em: %s", EXTENSAO, PASTA_INPUT)
        return

    log.info("Arquivos encontrados: %d", len(arquivos))

    movidos         = 0
    pulados         = 0
    ignorados       = 0
    novos_registros: list[dict] = []

    for arquivo in arquivos:
        # filtra arquivos abaixo do tamanho minimo
        tamanho = arquivo.stat().st_size
        if tamanho < TAMANHO_MINIMO_BYTES:
            log.warning("Ignorado (muito pequeno, %d bytes): %s", tamanho, arquivo.name)
            ignorados += 1
            continue

        hash_id = gerar_hash(arquivo)

        if hash_id in hashes_existentes:
            log.warning("Ja registrado, pulando: %s", arquivo.name)
            pulados += 1
            continue

        pasta_destino = PASTA_OUTPUT / hash_id
        pasta_destino.mkdir(parents=True, exist_ok=True)

        # arquivo renomeado com o mesmo hash da pasta, mantendo a extensao
        destino_final = pasta_destino / f"{hash_id}{arquivo.suffix}"

        shutil.move(str(arquivo), str(destino_final))
        log.info("Movido: %s -> %s", arquivo.name, destino_final)

        novos_registros.append({
            "hash"            : hash_id,
            "nome_audio"      : arquivo.stem,
            "caminho_origem"  : str(arquivo),
            "caminho_destino" : str(destino_final),
        })
        movidos += 1

    if novos_registros:
        registrar_relacao(arquivo_relacao, novos_registros)

    log.info(
        "Concluido. Movidos: %d | Pulados: %d | Ignorados: %d | Relacao: %s",
        movidos, pulados, ignorados, arquivo_relacao,
    )


if __name__ == "__main__":
    main()
