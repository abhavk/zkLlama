#![no_main]
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::collections::HashMap;
use byteorder::{ReadBytesExt, LittleEndian};
use rand::prelude::*;
use memmap2::MmapOptions;
use std::slice;
sp1_zkvm::entrypoint!(main);
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

#[derive(Serialize, Deserialize)]
struct TransformerWeights {
    token_embedding_table: Vec<f32>,
    rms_att_weight: Vec<f32>,
    rms_ffn_weight: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    rms_final_weight: Vec<f32>,
    wcls: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct Transformer {
    config: Config,
    weights: TransformerWeights,
    data: Vec<u8>,
}

impl Transformer {
    pub fn new(checkpoint_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(checkpoint_path)?;
        
        // Read config
        let mut config_bytes = [0u8; std::mem::size_of::<Config>()];
        file.read_exact(&mut config_bytes)?;
        let config: Config = unsafe { std::mem::transmute(config_bytes) };

        // Memory map the file
        let data_map: memmap2::Mmap = unsafe { MmapOptions::new().map(&file)? };
        let data = data_map.to_vec();

        // Calculate offsets for different weight matrices
        let mut offset = std::mem::size_of::<Config>();
        
        let weights = unsafe {
            let ptr = data.as_ptr().add(offset) as *const f32;
            let token_embedding_table = slice::from_raw_parts(ptr, config.vocab_size as usize * config.dim as usize);
            offset += config.vocab_size as usize * config.dim as usize * std::mem::size_of::<f32>();    

            let ptr = data.as_ptr().add(offset) as *const f32;
            let rms_att_weight = slice::from_raw_parts(ptr, config.n_layers as usize * config.dim as usize);
            offset += config.n_layers as usize * config.dim as usize * std::mem::size_of::<f32>();

            let ptr = data.as_ptr().add(offset) as *const f32;
            let wq = slice::from_raw_parts(ptr, config.n_layers as usize * config.dim as usize * config.dim as usize);
            offset += config.n_layers as usize * config.dim as usize * config.dim as usize * std::mem::size_of::<f32>();

            let ptr = data.as_ptr().add(offset) as *const f32;
            let wk = slice::from_raw_parts(ptr, config.n_layers as usize * config.dim as usize * config.dim as usize);
            offset += config.n_layers as usize * config.dim as usize * config.dim as usize * std::mem::size_of::<f32>();

            let ptr = data.as_ptr().add(offset) as *const f32;
            let wv = slice::from_raw_parts(ptr, config.n_layers as usize * config.dim as usize * config.dim as usize);
            offset += config.n_layers as usize * config.dim as usize * config.dim as usize * std::mem::size_of::<f32>();

            let ptr = data.as_ptr().add(offset) as *const f32;
            let wo = slice::from_raw_parts(ptr, config.n_layers as usize * config.dim as usize * config.dim as usize);
            offset += config.n_layers as usize * config.dim as usize * config.dim as usize * std::mem::size_of::<f32>();

            let ptr = data.as_ptr().add(offset) as *const f32;
            let rms_ffn_weight = slice::from_raw_parts(ptr, config.n_layers as usize * config.dim as usize);
            offset += config.n_layers as usize * config.dim as usize * std::mem::size_of::<f32>();

            let ptr = data.as_ptr().add(offset) as *const f32;
            let w1 = slice::from_raw_parts(ptr, config.n_layers as usize * config.dim as usize * config.hidden_dim as usize);
            offset += config.n_layers as usize * config.dim as usize * config.hidden_dim as usize * std::mem::size_of::<f32>();

            let ptr = data.as_ptr().add(offset) as *const f32;
            let w2 = slice::from_raw_parts(ptr, config.n_layers as usize * config.hidden_dim as usize * config.dim as usize);
            offset += config.n_layers as usize * config.hidden_dim as usize * config.dim as usize * std::mem::size_of::<f32>();

            let ptr = data.as_ptr().add(offset) as *const f32;
            let w3 = slice::from_raw_parts(ptr, config.n_layers as usize * config.dim as usize * config.hidden_dim as usize);
            offset += config.n_layers as usize * config.dim as usize * config.hidden_dim as usize * std::mem::size_of::<f32>();

            let ptr = data.as_ptr().add(offset) as *const f32;
            let rms_final_weight = slice::from_raw_parts(ptr, config.dim as usize);
            offset += config.dim as usize * std::mem::size_of::<f32>();

            // Skip freq_cis_real and freq_cis_imag
            offset += config.seq_len as usize * config.dim as usize * 2 * std::mem::size_of::<f32>();
            let ptr = data.as_ptr().add(offset) as *const f32;
            let wcls = if ptr as *const f32 == token_embedding_table.as_ptr() {
                token_embedding_table  // shared weights
            } else {
                slice::from_raw_parts(ptr, (config.vocab_size * config.dim) as usize)
            };

            TransformerWeights {
                token_embedding_table: token_embedding_table.to_vec(),
                rms_att_weight: rms_att_weight.to_vec(),
                rms_ffn_weight: rms_ffn_weight.to_vec(),
                wq: wq.to_vec(),
                wk: wk.to_vec(),
                wv: wv.to_vec(),
                wo: wo.to_vec(),
                w1: w1.to_vec(),
                w2: w2.to_vec(),
                w3: w3.to_vec(),
                rms_final_weight: rms_final_weight.to_vec(),
                wcls: wcls.to_vec(),
            }
        };

        Ok(Transformer { config, weights, data })
    }


    fn forward(&mut self, token: i32, pos: i32) -> Vec<f32> {
        // This is still a placeholder. In a full implementation, this would
        // perform the actual forward pass of the transformer.
        println!("Forward pass with token {} at position {}", token, pos);
        
        // For now, we'll return random logits
        let mut rng = rand::thread_rng();
        (0..self.config.vocab_size).map(|_| rng.gen()).collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    vocab_map: HashMap<String, usize>,
    max_token_length: usize,
}

impl Tokenizer {
    pub fn new(tokenizer_path: &str) -> Result<Self, std::io::Error> {
        let mut file = File::open(tokenizer_path)?;
        let mut max_token_length = [0u8; 4];
        file.read_exact(&mut max_token_length)?;
        let max_token_length = i32::from_le_bytes(max_token_length) as usize;

        let mut vocab = Vec::new();
        let mut vocab_scores = Vec::new();
        let mut vocab_map = HashMap::new();

        loop {
            let mut score_bytes = [0u8; 4];
            if file.read_exact(&mut score_bytes).is_err() {
                break;
            }
            let score = f32::from_le_bytes(score_bytes);

            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            let len = i32::from_le_bytes(len_bytes) as usize;

            let mut word = vec![0u8; len];
            file.read_exact(&mut word)?;
            let word = String::from_utf8(word).expect("Invalid UTF-8");

            vocab_map.insert(word.clone(), vocab.len());
            vocab.push(word);
            vocab_scores.push(score);
        }

        Ok(Tokenizer {
            vocab,
            vocab_scores,
            vocab_map,
            max_token_length,
        })
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<i32> {
        let mut tokens = Vec::new();
        if bos {
            tokens.push(1); // BOS token
        }

        let mut pos = 0;
        while pos < text.len() {
            let mut best_len = 0;
            let mut best_token = 0;

            for (token, word) in self.vocab.iter().enumerate() {
                if text[pos..].starts_with(word) && word.len() > best_len {
                    best_len = word.len();
                    best_token = token as i32;
                }
            }

            if best_len == 0 {
                // Unknown token, use byte fallback
                best_len = 1;
                best_token = (text.as_bytes()[pos] as i32) + 3; // +3 because the first 3 tokens are special
            }

            tokens.push(best_token);
            pos += best_len;
        }

        if eos {
            tokens.push(2); // EOS token
        }

        tokens
    }

    fn decode(&self, token: i32) -> &str {
        if token < 3 || token >= (self.vocab.len() as i32) {
            // Special tokens or out of range
            return "";
        }
        &self.vocab[token as usize]
    }
}

fn generate(transformer: &mut Transformer, tokenizer: &Tokenizer, sampler: &mut Sampler, prompt: &str, steps: i32) -> Vec<u8> {
    let tokens = tokenizer.encode(prompt, true, false);
    let mut pos = 0;
    let mut bytes = Vec::new();
    for &token in &tokens {
        let _logits = transformer.forward(token, pos);
        pos += 1;
        print!("{}", tokenizer.decode(token));
        bytes.push(token as u8);
    }
    for _ in pos..steps {
        let logits = transformer.forward(tokens[tokens.len() - 1], pos);
        let next_token = sampler.sample(&logits) as i32;
        print!("{}", tokenizer.decode(next_token));
        bytes.push(next_token as u8);
        pos += 1;
    }
    println!();
    bytes
}

struct Sampler {
    vocab_size: usize,
    temperature: f32,
    topp: f32,
    rng: rand::prelude::StdRng,
}

impl Sampler {
    fn new(vocab_size: usize, temperature: f32, topp: f32, seed: u64) -> Self {
        Sampler {
            vocab_size,
            temperature,
            topp,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn sample(&mut self, logits: &[f32]) -> usize {
        if self.temperature == 0.0 {
            // Greedy sampling
            logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap()
        } else {
            // Apply temperature
            let scaled_logits: Vec<f32> = logits.iter()
                .map(|&l| l / self.temperature)
                .collect();
            
            // Compute softmax
            let max_logit = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = scaled_logits.iter()
                .map(|&l| (l - max_logit).exp())
                .collect();
            let sum_exp_logits: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter()
                .map(|&e| e / sum_exp_logits)
                .collect();

            if self.topp <= 0.0 || self.topp >= 1.0 {
                // Regular temperature sampling
                let mut cumulative_prob = 0.0;
                let sample = self.rng.gen::<f32>();
                for (i, &p) in probs.iter().enumerate() {
                    cumulative_prob += p;
                    if sample < cumulative_prob {
                        return i;
                    }
                }
                probs.len() - 1 // Fallback to last token if rounding errors occur
            } else {
                // Top-p (nucleus) sampling
                let mut sorted_probs: Vec<(usize, f32)> = probs.iter()
                    .cloned()
                    .enumerate()
                    .collect();
                sorted_probs.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
                
                let mut cumulative_prob = 0.0;
                let mut nucleus = Vec::new();
                for (idx, p) in sorted_probs {
                    if cumulative_prob >= self.topp {
                        break;
                    }
                    nucleus.push((idx, p));
                    cumulative_prob += p;
                }

                let nucleus_prob: f32 = nucleus.iter().map(|&(_, p)| p).sum();
                let sample = self.rng.gen::<f32>() * nucleus_prob;
                let mut cumulative_prob = 0.0;
                for (idx, p) in &nucleus {
                    cumulative_prob += *p;
                    if sample < cumulative_prob {
                        return *idx;
                    }
                }
                nucleus.last().map(|&(idx, _)| idx).unwrap_or(0) // Fallback to first token if empty
            }
        }
    }
}

pub fn main() {
    let n = sp1_zkvm::io::read::<u32>();
    println!("Reading n inside the main function: {}", n);
    let checkpoint_path = "stories15M.bin";
    let tokenizer_path = "tokenizer.bin";
    let mut transformer = Transformer::new(checkpoint_path).unwrap();
    let tokenizer = Tokenizer::new(tokenizer_path).unwrap();
    let mut sampler = Sampler::new(
        transformer.config.vocab_size as usize,
        0.8, // temperature
        0.9, // top-p
        42,  // random seed
    );
    
    let prompt = "Once upon a time";
    let bytes = generate(&mut transformer, &tokenizer, &mut sampler, prompt, 10);

    sp1_zkvm::io::commit_slice(&bytes);
    // Print some information about the weights
    println!("Llama 2 Rust implementation loaded successfully");
    println!("Model configuration:");
    println!("  dim: {}", transformer.config.dim);
    println!("  hidden_dim: {}", transformer.config.hidden_dim);
    println!("  n_layers: {}", transformer.config.n_layers);
    println!("  n_heads: {}", transformer.config.n_heads);
    println!("  n_kv_heads: {}", transformer.config.n_kv_heads);
    println!("  vocab_size: {}", transformer.config.vocab_size);
    println!("  seq_len: {}", transformer.config.seq_len);

    println!("\nWeight shapes:");
    println!("  Token embedding table: {}", transformer.weights.token_embedding_table.len());
    println!("  WQ: {}", transformer.weights.wq.len());
    println!("  WK: {}", transformer.weights.wk.len());
    println!("  WV: {}", transformer.weights.wv.len());
    println!("  WO: {}", transformer.weights.wo.len());
    println!("  W1: {}", transformer.weights.w1.len());
    println!("  W2: {}", transformer.weights.w2.len());
    println!("  W3: {}", transformer.weights.w3.len());

    // Verify some calculations
    println!("\nVerifications:");
    println!("  vocab_size * dim = {}", transformer.config.vocab_size as usize * transformer.config.dim as usize);
    println!("  n_layers * dim * dim = {}", transformer.config.n_layers as usize * transformer.config.dim as usize * transformer.config.dim as usize);


}