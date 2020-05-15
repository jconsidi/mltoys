# mltoys/test.py


def test_models(model_class, factory):
    test_losses = []
    for instance in factory.get_instances(count=10):
        model = model_class(
            columns=instance.columns,
            feature_columns=instance.feature_columns,
            target_columns=instance.target_columns,
            loss_function=instance.loss_function,
        )

        model.fit(instance.training_data)

        test_loss = instance.calculate_loss(model.predict(instance.test_data))
        test_losses.append(test_loss)

    print(
        f"{factory.__class__.__name__} : min/mean/max = {min(test_losses):.4f}/{sum(test_losses)/len(test_losses):.4f}/{max(test_losses):.4f} ({instance.loss_function})"
    )
