from ejt import Config, EJTPipeline


def main():
    config = Config(
        model_name="",
        template_path="",
        query_path="",
        template_column="",
        query_column="",
        output_dir="",
        output_filename="",
    )

    pipeline = EJTPipeline(config)
    pipeline.run()

    print("EJT generation finished.")


if __name__ == "__main__":
    main()
