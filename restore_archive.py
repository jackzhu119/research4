#!/usr/bin/env python3
"""Download and verify the public multipart ML evidence archive; no training.

By default, retain separate verified shards. --verify-gzip streams across shards
without reassembly. Use --commit with the archive branch's exact commit SHA for
an immutable source. Python 3.9+; standard library only; no GitHub token needed.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_REPO = 'jackzhu119/research4'
DEFAULT_REF = 'research/background-study-1360000-20261115-archive'
BLOCK = 1024 * 1024


def digest_file(path):
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while True:
            data = stream.read(BLOCK)
            if not data:
                break
            size += len(data)
            h.update(data)
    return size, h.hexdigest()


def safe_basename(value):
    if not isinstance(value, str) or not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError('Expected a plain filename: ' + repr(value))
    return value


def load_manifest(path):
    with path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    shards = manifest.get('shards')
    if not isinstance(shards, list) or not shards:
        raise ValueError('Manifest has no shards')
    shards = sorted(shards, key=lambda row: int(row['part']))
    names = set()
    for number, row in enumerate(shards, 1):
        if int(row['part']) != number:
            raise ValueError('Shard part numbers must be consecutive from 1')
        name = safe_basename(row['name'])
        if name in names:
            raise ValueError('Duplicate shard name: ' + name)
        names.add(name)
        if not isinstance(row['bytes'], int) or isinstance(row['bytes'], bool) or row['bytes'] <= 0:
            raise ValueError('Invalid byte count: ' + name)
        if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid SHA256: ' + name)
    if manifest.get('shard_count', len(shards)) != len(shards):
        raise ValueError('Manifest shard_count differs from shard list')
    if manifest.get('compressed_archive_bytes', sum(row['bytes'] for row in shards)) != sum(row['bytes'] for row in shards):
        raise ValueError('Manifest compressed byte total differs from shard list')
    manifest['shards'] = shards
    safe_basename(manifest.get('archive_basename', 'ml-evidence-20261115.tar.gz'))
    return manifest


def raw_url(repo, ref, name):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Repository must be owner/name')
    if not ref or any(ch in ref for ch in '\r\n'):
        raise ValueError('Invalid Git ref')
    return 'https://raw.githubusercontent.com/' + repo + '/' + urllib.parse.quote(ref, safe='') + '/shards/' + urllib.parse.quote(name, safe='')


def matches(path, row):
    if path.is_symlink():
        raise ValueError('Refusing symlink: ' + str(path))
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError('Expected regular file: ' + str(path))
    return path.stat().st_size == row['bytes'] and digest_file(path) == (row['bytes'], row['sha256'])


def download(url, destination, row):
    temporary = destination.with_name(destination.name + '.partial')
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise ValueError('Refusing non-regular temporary path: ' + str(temporary))
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'research4-ml-archive-restore/1'})
            h = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as stream:
                while True:
                    data = response.read(BLOCK)
                    if not data:
                        break
                    count += len(data)
                    if count > row['bytes']:
                        raise ValueError('Download exceeds manifest byte count: ' + row['name'])
                    h.update(data)
                    stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if count != row['bytes'] or h.hexdigest() != row['sha256']:
                raise ValueError('Downloaded size/SHA256 mismatch: ' + row['name'])
            os.replace(temporary, destination)
            return
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == 4:
                raise
        finally:
            if temporary.exists():
                temporary.unlink()
        time.sleep(min(16, 2 ** attempt))


class JoinedShards(io.RawIOBase):
    """Read multiple files as one non-seekable stream, without concatenating."""
    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.closed:
            raise ValueError('Stream is closed')
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open('rb')
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def verify_gzip(paths):
    count = 0
    with io.BufferedReader(JoinedShards(paths), buffer_size=BLOCK) as joined:
        with gzip.GzipFile(fileobj=joined, mode='rb') as stream:
            while True:
                data = stream.read(BLOCK)
                if not data:
                    break
                count += len(data)
    return count


def reassemble(paths, destination):
    if destination.exists() or destination.is_symlink():
        raise ValueError('Reassembly destination already exists: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    if temporary.exists() or temporary.is_symlink():
        raise ValueError('Reassembly temporary path already exists: ' + str(temporary))
    try:
        with temporary.open('xb') as output:
            for path in paths:
                with path.open('rb') as stream:
                    shutil.copyfileobj(stream, output, length=BLOCK)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_extracted(root, member_manifest, expected_manifest_sha):
    root = root.resolve()
    if expected_manifest_sha and digest_file(member_manifest)[1] != expected_manifest_sha:
        raise ValueError('Member manifest SHA256 mismatch')
    count = 0
    with gzip.open(member_manifest, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            member = PurePosixPath(row['path'])
            if member.is_absolute() or '..' in member.parts or '\\' in row['path']:
                raise ValueError('Unsafe member path: ' + row['path'])
            candidate = root.joinpath(*member.parts)
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
                raise ValueError('Unsafe extracted member: ' + row['path'])
            if not candidate.is_file() or digest_file(candidate) != (row['bytes'], row['sha256']):
                raise ValueError('Extracted member size/SHA256 mismatch: ' + row['path'])
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=Path('ARCHIVE_MANIFEST.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('ml_archive_download'))
    parser.add_argument('--commit', help='Exact archive commit SHA recommended; defaults to manifest archive_commit or archive branch')
    parser.add_argument('--repo', help='Defaults to manifest repo or jackzhu119/research4')
    parser.add_argument('--offline', action='store_true', help='Only verify existing local shards; never download')
    parser.add_argument('--reassemble', action='store_true', help='Create an additional full tar.gz copy after shard verification')
    parser.add_argument('--archive-path', type=Path, help='Reassembled filename; only used with --reassemble')
    parser.add_argument('--verify-gzip', action='store_true', help='Drain gzip to EOF and validate CRC directly across shards')
    parser.add_argument('--verify-extracted', type=Path, metavar='DIRECTORY', help='Verify extracted files relative to the parent of background_study_20261110/')
    parser.add_argument('--member-manifest', type=Path, help='Defaults to DIRECTORY/background_study_20261110/archive_members.jsonl.gz')
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    repo = args.repo or manifest.get('repo', DEFAULT_REPO)
    ref = args.commit or manifest.get('archive_commit') or DEFAULT_REF
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, row in enumerate(manifest['shards'], 1):
        path = args.output_dir / row['name']
        if not matches(path, row):
            if args.offline:
                raise ValueError('Missing or mismatched local shard: ' + str(path))
            download(raw_url(repo, ref, row['name']), path, row)
        paths.append(path)
        print(f'Verified {number}/{len(manifest["shards"])} {row["name"]}', flush=True)
    result = {'shards_verified': len(paths), 'compressed_bytes': sum(row['bytes'] for row in manifest['shards']), 'source_ref': ref, 'reassembled': False}
    if args.verify_gzip:
        result['gzip_crc_verified'] = True
        result['uncompressed_tar_bytes'] = verify_gzip(paths)
    if args.reassemble:
        archive = args.archive_path or args.output_dir / manifest.get('archive_basename', 'ml-evidence-20261115.tar.gz')
        reassemble(paths, archive)
        result.update(reassembled=True, archive_path=str(archive))
    if args.verify_extracted:
        member_manifest = args.member_manifest or args.verify_extracted / 'background_study_20261110/archive_members.jsonl.gz'
        result['extracted_members_verified'] = verify_extracted(args.verify_extracted, member_manifest, manifest.get('archive_members_manifest_sha256'))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, gzip.BadGzipFile, EOFError) as error:
        print('Archive restore failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
