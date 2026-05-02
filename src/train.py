import torch

from src.gpt.generate_text import text_to_token_ids, generate_text, token_ids_to_text

def calculate_loss_batch(input_batch, target_batch, model):
  logits = model(input_batch)
  loss = torch.nn.functional.cross_entropy(
      logits.flatten(0, 1), target_batch.flatten()
  )
  return loss

def calculate_loss_loader(data_loader, model):
  total_loss = 0.
  
  for (input_batch, target_batch) in data_loader:
    loss = calculate_loss_batch(
      input_batch, target_batch, model
    )
    total_loss += loss.item()

  return total_loss / len(data_loader)

def evaluate_model(model, train_loader, val_loader):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(
            train_loader, model
        )
        val_loss = calculate_loss_loader(
            val_loader, model
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model(model, train_loader, val_loader,
               optimizer, num_epochs, eval_freq, 
               start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calculate_loss_batch(
                input_batch, target_batch, model
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, start_context
        )
    return train_losses, val_losses, track_tokens_seen
