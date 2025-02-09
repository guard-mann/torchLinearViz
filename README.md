# torchLinearViz

![demo image](./image/overview.png)

# demo Code (from ./tests/demoMNIST.py)
```# torchLinearViz初期化
torchlinearviz = TorchLinearViz(model)

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    torchlinearviz.update(model, images)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

torchlinearviz.end()``````


