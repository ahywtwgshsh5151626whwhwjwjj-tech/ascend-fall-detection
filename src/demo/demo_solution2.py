from core.solution2_engine import Solution2Engine


def main():
    print("[Solution2] Demo start")

    engine = Solution2Engine()
    engine.init_input()
    result = engine.run_once()
    engine.output_result(result)

    print("[Solution2] Demo end")


if __name__ == "__main__":
    main()
