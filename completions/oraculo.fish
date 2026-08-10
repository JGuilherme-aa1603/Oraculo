# Completions do fish para o wrapper `oraculo`.
# Instalar: ln -s "$PWD/completions/oraculo.fish" ~/.config/fish/completions/

# Sem subcomando ainda digitado.
function __oraculo_sem_comando
    set -l tokens (commandline -opc)
    return (test (count $tokens) -le 1; and echo 0; or echo 1)
end

complete -c oraculo -f

complete -c oraculo -n __oraculo_sem_comando -a transcrever \
    -d "Transcreve um arquivo de áudio"
complete -c oraculo -n __oraculo_sem_comando -a ajuda \
    -d "Lista os comandos do Oráculo"

# Depois de `transcrever`: arquivos de mídia e a flag de salvar.
complete -c oraculo -n "__fish_seen_subcommand_from transcrever" -F \
    -k -a "(__fish_complete_suffix .ogg .opus .mp3 .m4a .wav .flac .mp4 .webm)"
complete -c oraculo -n "__fish_seen_subcommand_from transcrever" \
    -s s -l salvar -d "Grava a transcrição em Markdown"
