#compdef train eval
# Zsh completion for UniLab commands installed in the active Conda environment.

_unilab_complete() {
    if [[ ${words[1]} != "train" && ${words[1]} != "eval" ]]; then
        return 0
    fi

    local script_path repo_root
    script_path="${${(%):-%x}:A}"
    repo_root="${script_path:h:h:h}"

    local output
    output="$(
        unilab-complete --cword "$((CURRENT - 1))" -- "${words[@]}" 2>/dev/null \
            || PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" python -m unilab.tools.completion --cword "$((CURRENT - 1))" -- "${words[@]}" 2>/dev/null
    )" || return 0

    if [[ -z "$output" ]]; then
        if (( CURRENT <= 2 )); then
            _files
        fi
        return 0
    fi

    local -a candidates
    candidates=("${(@f)output}")
    compadd -- "${candidates[@]}"
}

compdef _unilab_complete train eval
