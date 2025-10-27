"""
Command-line interface for the trading bot.

This module provides CLI commands using Typer for managing and running
the trading bot, including live trading, backtesting, and analysis.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="bot",
    help="AI-Enhanced Cryptocurrency Trading Bot",
    add_completion=False,
)

console = Console()


@app.command()
def version() -> None:
    """Display bot version information"""
    rprint("[bold green]AI-Enhanced Cryptocurrency Trading Bot[/bold green]")
    rprint("Version: [cyan]0.1.0[/cyan]")
    rprint(f"Python: [cyan]{sys.version.split()[0]}[/cyan]")


@app.command()
def backtest(
    data_file: Annotated[
        Path,
        typer.Option(
            "--data",
            "-d",
            help="Path to historical data file (CSV or Parquet)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Strategy to backtest (e.g., 'rsi', 'ma_crossover')",
        ),
    ] = "rsi",
    symbol: Annotated[
        str,
        typer.Option(
            "--symbol",
            help="Trading symbol (e.g., 'BTCUSDT')",
        ),
    ] = "BTCUSDT",
    initial_capital: Annotated[
        float,
        typer.Option(
            "--capital",
            "-c",
            help="Initial capital for backtesting",
        ),
    ] = 10000.0,
    commission: Annotated[
        float,
        typer.Option(
            "--commission",
            help="Trading commission rate (e.g., 0.001 for 0.1%)",
        ),
    ] = 0.001,
    slippage: Annotated[
        float,
        typer.Option(
            "--slippage",
            help="Slippage factor (e.g., 0.001 for 0.1%)",
        ),
    ] = 0.001,
    start_date: Annotated[
        str | None,
        typer.Option(
            "--start",
            help="Start date for backtesting (YYYY-MM-DD)",
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(
            "--end",
            help="End date for backtesting (YYYY-MM-DD)",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results",
        ),
    ] = None,
    replay_mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="Replay mode: 'bar' or 'tick'",
        ),
    ] = "bar",
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to strategy configuration file (YAML/JSON)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """
    Run backtesting simulation on historical data.

    Examples:
        # Basic backtest with RSI strategy
        bot backtest --data data/BTCUSDT.csv --strategy rsi

        # Backtest with custom parameters
        bot backtest -d data/BTCUSDT.parquet -s ma_crossover --capital 50000 --commission 0.0015

        # Backtest with date range
        bot backtest -d data/BTCUSDT.csv -s rsi --start 2023-01-01 --end 2023-12-31

        # Tick-by-tick replay mode
        bot backtest -d data/BTCUSDT.csv -s rsi --mode tick
    """
    from bot.data.backtesting import BacktestConfig, BacktestEngine

    try:
        # Parse dates if provided
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        # Create output directory if specified
        output_dir = output or Path("backtest_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load strategy config if provided
        strategy_config = None
        if config_file:
            import json

            import yaml

            if config_file.suffix == ".json":
                with open(config_file) as f:
                    strategy_config = json.load(f)
            else:  # YAML
                with open(config_file) as f:
                    strategy_config = yaml.safe_load(f)

        # Create backtest configuration
        config = BacktestConfig(
            data_file=str(data_file),
            strategy_name=strategy,
            symbol=symbol,
            initial_capital=initial_capital,
            commission_rate=commission,
            slippage_factor=slippage,
            start_date=start_dt,
            end_date=end_dt,
            replay_mode=replay_mode,
            strategy_config=strategy_config,
        )

        console.print("\n[bold cyan]Starting Backtest[/bold cyan]\n")
        console.print(f"[yellow]Data File:[/yellow] {data_file}")
        console.print(f"[yellow]Strategy:[/yellow] {strategy}")
        console.print(f"[yellow]Symbol:[/yellow] {symbol}")
        console.print(f"[yellow]Initial Capital:[/yellow] ${initial_capital:,.2f}")
        console.print(f"[yellow]Commission:[/yellow] {commission * 100:.2f}%")
        console.print(f"[yellow]Slippage:[/yellow] {slippage * 100:.2f}%")
        console.print(f"[yellow]Replay Mode:[/yellow] {replay_mode}")
        if start_dt:
            console.print(f"[yellow]Start Date:[/yellow] {start_dt.date()}")
        if end_dt:
            console.print(f"[yellow]End Date:[/yellow] {end_dt.date()}")
        console.print()

        # Run backtest
        with console.status("[bold green]Running backtest..."):
            engine = BacktestEngine(config)
            results = engine.run()

        # Display results
        console.print("\n[bold green]Backtest Complete![/bold green]\n")

        # Summary table
        table = Table(title="Performance Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="yellow", justify="right")

        metrics = results.metrics
        table.add_row("Total Return", f"{metrics['total_return']:.2%}")
        table.add_row("Annual Return", f"{metrics['annual_return']:.2%}")
        table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        table.add_row("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
        table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        table.add_row("Total Trades", str(metrics["total_trades"]))
        table.add_row("Win Rate", f"{metrics['win_rate']:.2%}")
        table.add_row("Profit Factor", f"{metrics['profit_factor']:.2f}")
        table.add_row("Average Win", f"${metrics['avg_win']:.2f}")
        table.add_row("Average Loss", f"${metrics['avg_loss']:.2f}")
        table.add_row("Final Capital", f"${metrics['final_capital']:.2f}")

        console.print(table)

        # Save results
        report_file = (
            output_dir
            / f"backtest_{strategy}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        results.save_report(str(report_file))

        console.print(f"\n[green]Results saved to:[/green] {report_file}")
        console.print(f"[green]Trade log:[/green] {output_dir / 'trades.csv'}")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def live(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to bot configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    """
    Run the bot in live trading mode.

    This command starts the bot for live trading using the specified configuration.
    """
    console.print("[yellow]Live trading not yet implemented[/yellow]")
    console.print(f"Would load config from: {config}")


@app.command()
def paper(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to bot configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    """
    Run the bot in paper trading mode.

    This command starts the bot for paper trading (simulated) using the specified configuration.
    """
    console.print("[yellow]Paper trading not yet implemented[/yellow]")
    console.print(f"Would load config from: {config}")


def main() -> None:
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()
