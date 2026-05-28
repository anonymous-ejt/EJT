import os

from ejt import Config, EJTPipeline


def main():
    config = Config(
        model_name=os.getenv("OPENAI_MODEL", "gpt-5.4"),
        template_path="data/Example_Template_for_EJT.csv",
        query_path="data/Example_Query_for_EJT.csv",
        template_column="prompt",
        query_column="Query",
        output_dir="outputs",
        output_filename="ejt_results.csv",
    )

    pipeline = EJTPipeline(config)
    pipeline.run()

    print("EJT generation finished.")


if __name__ == "__main__":
    main()
