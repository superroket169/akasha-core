use akasha_core::tokenizer::Tokenizer;

fn main() {
    let mut tk = Tokenizer::new();
    const SIZE: u32 = 50304;

    let egitim_verisi = "Hey rome, akasha is going rome door ise so many good bro rome rome rome rome rome rome there, make yourself sure! There was no way out. The system is done for. Why are tokens used for performance. My life is ruined. How does does does akasha work. ";

    tk.train(egitim_verisi, SIZE);

    let encode_edilmis = tk.encode("door");
    println!("Sayılara dönüştü: {:?}", encode_edilmis);

    let decode_edilmis = tk.decode(&encode_edilmis);
    println!("Geri metne dönüştü: {}", decode_edilmis);
}
