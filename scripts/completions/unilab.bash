# Bash completion for UniLab commands installed in the active Conda environment.

_unilab_complete() {
    COMPREPLY=()

    if [[ ${COMP_WORDS[0]} != "train" && ${COMP_WORDS[0]} != "eval" ]]; then
        return 0
    fi

    local script_dir repo_root
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    repo_root="$(cd -- "$script_dir/../.." && pwd -P)"

    local candidates
    if ! mapfile -t candidates < <(
        unilab-complete --cword "$COMP_CWORD" -- "${COMP_WORDS[@]}" 2>/dev/null \
            || PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" python -m unilab.tools.completion --cword "$COMP_CWORD" -- "${COMP_WORDS[@]}" 2>/dev/null
    ); then
        return 0
    fi

    if [[ ${#candidates[@]} -eq 0 ]]; then
        if [[ $COMP_CWORD -le 1 ]]; then
            compopt -o default -o bashdefault 2>/dev/null || true
            return 1
        fi
        return 0
    fi

    COMPREPLY=("${candidates[@]}")
}

complete -o default -o bashdefault -F _unilab_complete train eval
