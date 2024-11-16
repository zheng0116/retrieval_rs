case $1 in 
"clean")
    cargo clean
    rm cargo.lock
    ;;
"format")
    cargo fmt 
    cargo clippy
    ;;
"build")
    cargo build --release
    mv target/release/Retrieval_rs bin/
    ;;
"search")
    cargo run -- --image-dir "./quary/" --query "a dog"
    ;;
esac
